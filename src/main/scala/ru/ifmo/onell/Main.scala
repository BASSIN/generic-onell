package ru.ifmo.onell

import ru.ifmo.onell.algorithm.{OnePlusLambdaLambdaGA, OnePlusOneEA, RLS}
import ru.ifmo.onell.problem.{OneMax, OneMaxPerm}

object Main {
  private def usage(): Nothing = {
    System.err.println("Usage: Main <bits:om:simple | perm:om:simple>")
    sys.exit()
  }

  private def bitsOneMaxSimple(): Unit = {
    val algorithms = Seq(
      "RLS" -> RLS,
      "(1+1) EA" -> OnePlusOneEA,
      "(1+(λ,λ)) GA" -> new OnePlusLambdaLambdaGA(OnePlusLambdaLambdaGA.defaultAdaptiveLambda),
    )
    for (n <- 1000 to 10000 by 1000) {
      println(s"n = $n:")
      val oneMax = new OneMax(n)
      for ((name, alg) <- algorithms) {
        val runs = IndexedSeq.fill(100)(alg.optimize(oneMax)).sorted
        println(f"  $name%12s: ${runs.sum.toDouble / runs.size}%9.2f (min = ${runs.head}%6d, max = ${runs.last}%6d)")
      }
      println()
    }
  }

  private def permOneMaxSimple(): Unit = {
    val algorithms = Seq(
      "RLS" -> RLS,
      "(1+1) EA" -> OnePlusOneEA,
      "(1+(λ,λ)) GA" -> new OnePlusLambdaLambdaGA(OnePlusLambdaLambdaGA.fixedLambda(10)),
    )
    for (n <- 100 to 1000 by 100) {
      println(s"n = $n:")
      val oneMaxPerm = new OneMaxPerm(n)
      for ((name, alg) <- algorithms) {
        val runs = IndexedSeq.fill(11)(alg.optimize(oneMaxPerm)).sorted
        println(f"  $name%12s: ${runs.sum.toDouble / runs.size}%9.2f (min = ${runs.head}%6d, max = ${runs.last}%6d)")
      }
      println()
    }
  }

  def main(args: Array[String]): Unit = {
    if (args.length == 0) {
      usage()
    } else args(0) match {
      case "bits:om:simple" => bitsOneMaxSimple()
      case "perm:om:simple" => permOneMaxSimple()
      case _ => usage()
    }
  }
}
