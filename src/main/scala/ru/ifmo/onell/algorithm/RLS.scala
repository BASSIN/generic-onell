package ru.ifmo.onell.algorithm

import java.util.concurrent.ThreadLocalRandom

import scala.annotation.tailrec
import scala.{specialized => sp}

import ru.ifmo.onell.{HasDeltaOperations, HasEvaluation, HasIncrementalEvaluation, HasIndividualOperations, Optimizer}
import ru.ifmo.onell.util.Specialization.{fitnessSpecialization => fsp}

/**
  * This is an implementation of randomized local search.
  */
object RLS extends Optimizer {
  final def optimize[I, @sp(fsp) F, D](fitness: HasEvaluation[I, F] with HasIncrementalEvaluation[I, D, F])
                                      (implicit deltaOps: HasDeltaOperations[D], indOps: HasIndividualOperations[I]): Int = {
    val problemSize = fitness.problemSize
    val individual = indOps.createStorage(problemSize)
    val delta = deltaOps.createStorage(problemSize)
    val rng = ThreadLocalRandom.current()

    @tailrec
    def iterate(f: F, soFar: Int): Int = if (fitness.isOptimalFitness(f)) soFar else {
      deltaOps.initializeDeltaWithGivenSize(delta, problemSize, 1, rng)
      val newF = fitness.applyDelta(individual, delta, f)
      if (fitness.compare(f, newF) <= 0) {
        iterate(newF, soFar + 1)
      } else {
        fitness.unapplyDelta(individual, delta)
        iterate(f, soFar + 1)
      }
    }

    indOps.initializeRandomly(individual, rng)
    iterate(fitness.evaluate(individual), 1)
  }
}
