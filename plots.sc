// scala 2.13.4

import $ivy. `org.carbonateresearch::picta:0.1.1`
import org.carbonateresearch.picta.render.Html.initNotebook // required to initialize jupyter notebook mode
initNotebook() // stops ugly output

import org.carbonateresearch.picta.IO._
import org.carbonateresearch.picta._

val filepath = getWorkingDirectory + "/metrics/lr.csv"
val data = readCSV(filepath)
val epochs = data("epoch").map(_.toInt)
val losses = data("loss").map(_.toDouble)

val series = XY(epochs, losses) asType SCATTER drawStyle LINES
val chart = Chart() addSeries series.setName("Learning loss") setTitle "Linear Regression Example: Loss vs. Epoch"
chart.plotInline

val data = readCSV(getWorkingDirectory + "/metrics/ann.csv")
val epochs = data("epoch").map(_.toInt)
val losses = data("loss").map(_.toDouble)
val accuracy = data("accuracy").map(_.toDouble)
val maxAccuracy = accuracy.max
val normAccuracy = accuracy.map(_ / maxAccuracy)
val maxLoss = losses.max
val normLoss = losses.map(_ / maxLoss)

val loss = XY(epochs, normLoss) asType SCATTER drawStyle LINES
val acc = XY(epochs, normAccuracy) asType SCATTER drawStyle LINES
val chart = 
  Chart() addSeries(
    loss.setName("Learning loss"), 
    acc.setName("Training Accuracy")
  ) setTitle "ANN Example: Loss vs. Accuracy vs. Epoch"     
chart.plotInline

val data = readCSV(s"$getWorkingDirectory/metrics/lr-surface.csv")
val w = data("w").map(_.toDouble)
val b = data("b").map(_.toDouble)
val loss = data("l").map(_.split(",").map(_.toDouble))
val surface = XYZ(x=w, y=b, z=loss.flatten, n=loss(0).length) asType SURFACE
val chart3 = Chart() addSeries surface setTitle "Loss Function Surface" addAxes(
  Axis(X, title = "w"), Axis(Y, title = "b"), Axis(Z, title = "loss")
)
chart3.plotInline

val contour = XYZ(x=w, y=b, z=loss.flatten, n=loss(0).length) asType CONTOUR
val chart1 = Chart().addSeries(contour).setTitle("Loss Contour")
             .addAxes(Axis(X, title = "w"), Axis(Y, title = "b"), Axis(Z, title = "loss"))
chart1.plotInline

import org.carbonateresearch.picta._
import org.carbonateresearch.picta.options._

val line = XYZ(x=w, y=b, z=loss.flatten) asType SCATTER3D drawStyle LINES
val chart4 = Chart() addSeries line setTitle "Line" setConfig(false, false)
chart4.plotInline