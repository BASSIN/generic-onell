package ru.ifmo.onell.algorithm

import java.util.concurrent.{ThreadLocalRandom => Random}

import ru.ifmo.onell._
import ru.ifmo.onell.algorithm.OnePlusLambdaLambdaGA.{Aux, ConstantTuning, CrossoverStrength, LambdaTuning, PopulationSizeRounding, defaultCrossoverStrength, defaultTuning, roundDownPopulationSize}
import ru.ifmo.onell.util.OrderedSet
import ru.ifmo.onell.util.Specialization.{changeSpecialization => csp, fitnessSpecialization => fsp}

import scala.annotation.tailrec
import scala.util.chaining._
import scala.{specialized => sp}

class OnePlusLambdaLambdaGAHT(lambdaTuning: Long => LambdaTuning,
                              constantTuning: ConstantTuning = defaultTuning,
                              populationRounding: PopulationSizeRounding = roundDownPopulationSize,
                              crossoverStrength: CrossoverStrength = defaultCrossoverStrength,
                              bePracticeAware: Boolean = true)
  extends Optimizer {
  override def optimize[I, @sp(fsp) F, @sp(csp) C]
  (fitness: Fitness[I, F, C],
   iterationLogger: IterationLogger[F])
  (implicit deltaOps: HasDeltaOperations[C], indOps: HasIndividualOperations[I]): Long = {
    val problemSize = fitness.problemSize
    val nChanges = fitness.numberOfChanges
    val lambdaP = lambdaTuning(fitness.changeIndexTypeToLong(nChanges))
    val rng = Random.current()
    val individual = indOps.createStorage(problemSize)
    val mutation, mutationBest, crossover, crossoverBest = deltaOps.createStorage(nChanges)
    val aux = new Aux[F]

    @tailrec
    def initMutation(expectedChange: Double): Int = {
      val change = deltaOps.initializeDeltaWithDefaultSize(mutation, nChanges, expectedChange, rng)
      if (change == 0 && bePracticeAware) initMutation(expectedChange) else change
    }

    def runMutationsHT(mutationPopSize: Int, baseFitness: F, change: Int): Array[(OrderedSet[C], F)] = {
      val mutantArr = new Array[(OrderedSet[C], F)](mutationPopSize)

      val currentFitness = fitness.evaluateAssumingDelta(individual, mutation, baseFitness)
      val currMutation = deltaOps.createStorage(nChanges)
      currMutation.copyFrom(mutation)
      mutantArr(0) = (currMutation, currentFitness)

      for (i <- 1 until mutationPopSize) {
        deltaOps.initializeDeltaWithGivenSize(mutation, nChanges, change, rng)
        val currentFitness = fitness.evaluateAssumingDelta(individual, mutation, baseFitness)
        val currMutation = deltaOps.createStorage(nChanges)
        currMutation.copyFrom(mutation)
        mutantArr(i) = (currMutation, currentFitness)
      }
      mutantArr.sortBy(_._2.asInstanceOf[Long])(Ordering[Long].reverse)
    }

    def calculateWeights(size: Int): Array[Double] = {
      val weights = new Array[Double](size)
      var weightsSum = 0.0
      for (i <- 1 to size) {
        weightsSum = weightsSum + math.pow(i, -2.5)
        weights(i - 1) = weightsSum
      }
      weights
    }

    def pickMutant(mutantArr: Array[(OrderedSet[C], F)], weightArr: Array[Double]): OrderedSet[C] = {
      val len = weightArr.length

      val r = rng.nextDouble(weightArr(len - 1))

      for (i <- 0 until len) {
        if (weightArr(i) >= r) {
          return mutantArr(i)._1
        }
      }
      mutantArr(0)._1
    }

    @tailrec
    def runPracticeAwareCrossoverHT(remaining: Int, baseFitness: F, expectedChange: Double, mutantDistance: Int,
                                    mutantArr: Array[(OrderedSet[C], F)], weightArr: Array[Double], result: Aux[F]): Unit = {
      if (remaining > 0) {

        //pick the mutant
        val mutant = pickMutant(mutantArr, weightArr)

        val size = deltaOps.initializeDeltaFromExisting(crossover, mutant, expectedChange, rng)
        if (size == 0) {
          // no bits from the child, skipping entirely
          runPracticeAwareCrossoverHT(remaining, baseFitness, expectedChange, mutantDistance, mutantArr, weightArr, result)
        } else {
          if (size != mutantDistance) {
            // if not all bits from the child, we shall evaluate the offspring, and if it is better, update the best
            val currFitness = fitness.evaluateAssumingDelta(individual, crossover, baseFitness)
            aux.incrementCalls()
            if (fitness.compare(aux.fitness, currFitness) <= 0) { // <= since we want to be able to overwrite parent
              crossoverBest.copyFrom(crossover)
              aux.fitness = currFitness
            }
          }
          runPracticeAwareCrossoverHT(remaining - 1, baseFitness, expectedChange, mutantDistance, mutantArr, weightArr, result)
        }
      }
    }

    def fitnessFromDistance(distance: Int, baseFitness: F, mutantDistance: Int, mutantFitness: F): F = {
      if (distance == 0)
        baseFitness
      else if (distance == mutantDistance)
        mutantFitness
      else
        fitness.evaluateAssumingDelta(individual, crossover, baseFitness)
    }

    @tailrec
    def runPracticeUnawareCrossoverImpl(remaining: Int, baseFitness: F, mutantFitness: F, bestFitness: F,
                                        expectedChange: Double, mutantDistance: Int): F = {
      if (remaining == 0) bestFitness else {
        val size = deltaOps.initializeDeltaFromExisting(crossover, mutationBest, expectedChange, rng)
        val newFitness = fitnessFromDistance(size, baseFitness, mutantDistance, mutantFitness)
        if (fitness.compare(bestFitness, newFitness) < 0) {
          crossoverBest.copyFrom(crossover)
          runPracticeUnawareCrossoverImpl(remaining - 1, baseFitness, mutantFitness, newFitness, expectedChange, mutantDistance)
        } else {
          runPracticeUnawareCrossoverImpl(remaining - 1, baseFitness, mutantFitness, bestFitness, expectedChange, mutantDistance)
        }
      }
    }

    def runPracticeUnawareCrossover(remaining: Int, baseFitness: F, mutantFitness: F, expectedChange: Double, mutantDistance: Int): F = {
      assert(remaining > 0)
      val size = deltaOps.initializeDeltaFromExisting(crossover, mutationBest, expectedChange, rng)
      val newFitness = fitnessFromDistance(size, baseFitness, mutantDistance, mutantFitness)
      crossoverBest.copyFrom(crossover)
      runPracticeUnawareCrossoverImpl(remaining - 1, baseFitness, mutantFitness, newFitness, expectedChange, mutantDistance)
    }

    @tailrec
    def iteration(f: F, evaluationsSoFar: Long): Long = if (fitness.isOptimalFitness(f)) evaluationsSoFar else {
      val lambda = lambdaP.lambda(rng)

      val mutationExpectedChanges = constantTuning.mutationProbabilityQuotient * lambda
      val mutationPopSize = math.max(1, populationRounding(lambda * constantTuning.firstPopulationSizeQuotient, rng))
      val crossoverPopSize = math.max(1, populationRounding(lambda * constantTuning.secondPopulationSizeQuotient, rng))


      val mutantDistance = initMutation(mutationExpectedChanges)
      if (mutantDistance == 0) {
        assert(!bePracticeAware)
        iteration(f, evaluationsSoFar + mutationPopSize + crossoverPopSize)
      } else {
        val mutantArr = runMutationsHT(mutationPopSize, f, mutantDistance)
        //val bestMutantFitness = runMutations(mutationPopSize, f, mutantDistance)
        val crossStrength = crossoverStrength(lambda, mutantDistance, constantTuning.crossoverProbabilityQuotient)
        if (bePracticeAware) {
          crossoverBest.copyFrom(mutantArr(0)._1)
          aux.initialize(mutantArr(0)._2)
          val weightArr = calculateWeights(mutationPopSize)
          runPracticeAwareCrossoverHT(
            crossoverPopSize, f,
            crossStrength,
            mutantDistance, mutantArr, weightArr, aux)
        } else {
          aux.initialize(runPracticeUnawareCrossover(crossoverPopSize, f, mutantArr(0)._2, crossStrength, mutantDistance))
          aux.incrementCalls(crossoverPopSize)
        }
        val bestCrossFitness = aux.fitness
        val crossEvs = aux.calls

        val fitnessComparison = fitness.compare(f, bestCrossFitness)
        if (fitnessComparison < 0) {
          lambdaP.notifyChildIsBetter()
        } else if (fitnessComparison > 0) {
          lambdaP.notifyChildIsWorse()
        } else {
          lambdaP.notifyChildIsEqual()
        }

        val iterationCost = mutationPopSize + crossEvs
        val nextFitness = if (fitnessComparison <= 0) {
          // maybe replace with silent application of delta
          fitness.applyDelta(individual, crossoverBest, f).tap(nf => assert(fitness.compare(bestCrossFitness, nf) == 0))
        } else f
        iterationLogger.logIteration(evaluationsSoFar + iterationCost, bestCrossFitness)
        iteration(nextFitness, evaluationsSoFar + iterationCost)
      }
    }

    indOps.initializeRandomly(individual, rng)
    val firstFitness = fitness.evaluate(individual)
    iterationLogger.logIteration(1, firstFitness)
    iteration(firstFitness, 1)
  }
}


