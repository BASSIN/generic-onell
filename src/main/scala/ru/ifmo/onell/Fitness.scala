package ru.ifmo.onell

import ru.ifmo.onell.util.OrderedSet
import ru.ifmo.onell.util.Specialization.{changeSpecialization => cs, fitnessSpecialization => fs}

/**
  * This trait encapsulates the knowledge for how to compute fitness,
  * and how to quickly compute the new fitness based on the old one
  * and the description of the change. The individual is assumed to be mutable.
  *
  * @tparam IndividualType the type of the individual.
  * @tparam FitnessType the type of the fitness value.
  * @tparam ChangeIndexType the type of the index of a single change (e.g. the index to flip a bit at).
  */
trait Fitness[IndividualType, @specialized(fs) FitnessType, @specialized(cs) ChangeIndexType] {
  /**
    * Evaluates the given individual.
    * @param individual the individual.
    * @return the fitness.
    */
  def evaluate(individual: IndividualType): FitnessType

  /**
    * Compares the given fitness values.
    * @param lhs the left-hand-side fitness.
    * @param rhs the right-hand-side fitness.
    * @return zero if fitness values are equal; negative if `lhs` is worse; positive if `lhs` is greater.
    */
  def compare(lhs: FitnessType, rhs: FitnessType): Int

  /**
    * Returns the problem size.
    * @return the problem size.
    */
  def problemSize: Int

  /**
    * Tests whether the given fitness is optimal.
    * @param fitness the fitness value.
    * @return `true` if the fitness is optimal, `false` otherwise.
    */
  def isOptimalFitness(fitness: FitnessType): Boolean

  /**
    * Returns the number of possible changes, given the problem size.
    * @param problemSize the problem size.
    * @return the number of possible changes.
    */
  def numberOfChangesForProblemSize(problemSize: Int): ChangeIndexType

  /**
    * Convert the change size type to long.
    * @param st the value of the change size type.
    * @return its long value.
    */
  def sizeTypeToLong(st: ChangeIndexType): Long

  /**
    * Applies the given delta to the given individual, while simultaneously recomputing the fitness.
    * @param ind the (mutable) individual.
    * @param delta the delta (the description of a change to the individual).
    * @param currentFitness the fitness of the individual before applying the change.
    * @return the new fitness after applying the change; the individual is also changed when the method returns.
    */
  def applyDelta(ind: IndividualType, delta: OrderedSet[ChangeIndexType], currentFitness: FitnessType): FitnessType

  /**
    * Unapplies the given delta to the given individual.
    * @param ind the (mutable) individual, which has previously experienced applying exactly the same delta.
    * @param delta the delta to be unapplied.
    */
  def unapplyDelta(ind: IndividualType, delta: OrderedSet[ChangeIndexType]): Unit

  /**
    * Evaluates the given individual, assuming the delta is applied.
    * The individual shall retain in the original state after applying this operation,
    * although it may be modified during the computation.
    *
    * @param ind the individual to measure assuming delta is applied.
    * @param delta the delta to be applied.
    * @param currentFitness the current fitness of the individual.
    * @return the new fitness after applying the change; the individual remains intact.
    */
  def evaluateAssumingDelta(ind: IndividualType, delta: OrderedSet[ChangeIndexType],
                            currentFitness: FitnessType): FitnessType = {
    val result = applyDelta(ind, delta, currentFitness)
    unapplyDelta(ind, delta)
    result
  }
}
