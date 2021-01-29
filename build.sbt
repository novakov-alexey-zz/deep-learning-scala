import Dependencies._

ThisBuild / scalaVersion := "2.13.4"
ThisBuild / version := "0.1.0-SNAPSHOT"
ThisBuild / organization := "io.github.novakov-alexey"
ThisBuild / organizationName := "novakov-alexey"

Global / onChangedBuildSource := ReloadOnSourceChanges

lazy val root = (project in file("."))
  .settings(
    name := "ann",
    libraryDependencies ++=
      Seq(scalaTest % Test)
  )
  .enablePlugins(ScalaNativePlugin)

import scala.scalanative.build._

nativeConfig ~= {
  _.withLTO(LTO.thin)
    .withMode(Mode.releaseFull)
    .withGC(GC.none)
}
