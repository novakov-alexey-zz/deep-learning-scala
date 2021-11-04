import Dependencies._

ThisBuild / scalaVersion := "3.1.0"
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
        "org.scala-lang.modules" % "scala-parallel-collections_3" % "1.0.3",
        scalaTest % Test
      )
  )

scalacOptions ++= Seq(
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
