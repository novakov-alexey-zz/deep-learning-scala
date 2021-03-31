import Dependencies._

ThisBuild / scalaVersion := "3.0.0-RC1"
ThisBuild / version := "0.1.0-SNAPSHOT"
ThisBuild / organization := "io.github.novakov-alexey"
ThisBuild / organizationName := "novakov-alexey"

Global / onChangedBuildSource := ReloadOnSourceChanges

lazy val root = (project in file("."))
  .settings(
    name := "ann",
    run / javaOptions += "-Xmx4G",
    fork := true,
    libraryDependencies ++=
      Seq(
        "org.scala-lang.modules" %% "scala-parallel-collections" % "1.0.0",
        scalaTest % Test
      )
  )

scalacOptions ++= {
  if (isDotty.value)
    Seq(
      "-encoding",
      "UTF-8",
      "-feature",
      "-unchecked",
      "-language:implicitConversions"
//      "-indent",
//      "-rewrite"
      // "-new-syntax",
      // "-Xfatal-warnings" will be added after the migration
    )
  else
    Seq(
      "-encoding",
      "UTF-8",
      "-feature",
      "-deprecation",
      "-language:implicitConversions",
      "-Xfatal-warnings",
      "-Wunused:imports,privates,locals",
      "-Wvalue-discard"
    )
}
