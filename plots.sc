// scala 2.13.4

import $ivy. `org.carbonateresearch::picta:0.1.1`
import org.carbonateresearch.picta.render.Html.initNotebook // required to initialize jupyter notebook mode
initNotebook() // stops standard output

import org.carbonateresearch.picta.IO._
import org.carbonateresearch.picta._

val metricsDir = getWorkingDirectory + "/../metrics"
val filepath = metricsDir + "/lr.csv"
val data = readCSV(filepath)
val epochs = data("epoch").map(_.toInt)
val losses = data("loss").map(_.toDouble)

val series = XY(epochs, losses).asType(SCATTER).drawStyle(LINES)
val chart = Chart().addSeries(series.setName("Learning loss")).setTitle("Linear Regression Example: Loss vs. Epoch")
chart.plotInline

val filepath = s"$metricsDir/datapoints.csv"
val data = readCSV(filepath)
val x = data("x").map(_.toDouble)
val y = data("y").map(_.toDouble)
val w = 0.6911375732835148
val b = 0.7800122918798839
def model(x: Double) = w * x + b
val m1 = Array(-0.1d, 1.3d)
val m2 = List(model(m1(0)), model(m1(1)))

//val marker = Marker() setSymbol SQUARE_OPEN setColor "red"
val inputData = XY(x, y) asType SCATTER setName "Input Data" drawStyle MARKERS //setMarker marker
val modelData = XY(m1.toList, m2) asType SCATTER setName "Model" // drawStyle MARKERS
val chart = Chart() addSeries(inputData, modelData) setTitle("Data points vs. Trained model")

chart.plotInline

val filepath = metricsDir + "/ann.csv"
val data = readCSV(filepath)
val epochs = data("epoch").map(_.toInt)
val losses = data("loss").map(_.toDouble)
val accuracy = data("accuracy").map(_.toDouble)
val maxAccuracy = accuracy.max
val normAccuracy = accuracy.map(_ / maxAccuracy)
val maxLoss = losses.max
val normLoss = losses.map(_ / maxLoss)

val loss = XY(epochs, losses) asType SCATTER drawStyle LINES
val acc = XY(epochs, accuracy) asType SCATTER drawStyle LINES
val lossChart = 
  Chart() addSeries(
    loss.setName("Learning loss"), 
    acc.setName("Training Accuracy")
  ) setTitle "ANN Example: Loss vs. Accuracy vs. Epoch"     
lossChart.plotInline

val data = readCSV(s"$metricsDir/adam-lr-surface.csv")
val w = data("w").map(_.toDouble).reverse
val b = data("b").map(_.toDouble).reverse
val loss = data("l").map(_.split(",").map(_.toDouble)).reverse
val surface = XYZ(x=w, y=b, z=loss.flatten, n=loss(0).length).asType(SURFACE).setColorBar("Loss", RIGHT_SIDE)

val gradientData = readCSV(s"$metricsDir/adam-gradient.csv")
val gw = gradientData("w").map(_.toDouble).reverse
val gb = gradientData("b").map(_.toDouble).reverse
val gLoss = gradientData("loss").map(_.toDouble).reverse
val gradient = XYZ(x=gw, y=gb, z=gLoss).asType(SCATTER3D).setName("Gradient").drawLinesMarkers

val surfaceChart = Chart()
    .addSeries(gradient,surface)
    .setTitle("Loss Function Surface")
    .setLegend(x = 0.5, y = -0.5, orientation = HORIZONTAL, xanchor = AUTO, yanchor = AUTO)
    .addAxes(Axis(X, title = "w"), Axis(Y, title = "b"), Axis(Z, title = "loss"))
surfaceChart.plotInline

import org.carbonateresearch.picta.options.Marker
import org.carbonateresearch.picta.SymbolShape._
import org.carbonateresearch.picta.options.AUTO

val contour = XYZ(x=w, y=b, z=loss.flatten, n=loss(0).length).asType(CONTOUR)
val adamdMarker = Marker().setColor("rgb(200,0,0)").setSymbol(SQUARE_OPEN)
val adamGradient = XY(x=gw, y=gb).asType(SCATTER).setName("Adam Gradient").setMarker(adamdMarker)
.drawLinesMarkers

val simpledGradientData = readCSV(s"$metricsDir/simplegd-gradient.csv")
val simpleGw = simpledGradientData("w").map(_.toDouble).reverse
val simpleGb = simpledGradientData("b").map(_.toDouble).reverse
val simpleGdmarker = Marker().setColor("rgb(0,200,0)").setSymbol(SQUARE_OPEN)
val simpleGDGradient = XY(x=simpleGw, y=simpleGb).asType(SCATTER)
    .setName("Classic Gradient Descent").setMarker(simpleGdmarker).drawLinesMarkers

val simpleGDAnimation = 
    (0 to simpleGw.length-1)
    .map(x => XY(simpleGw.take(x+1), simpleGb.take(x+1)) setName "Classic Gradient Descent")
    .toList

val adamAnimation = 
    (0 to gw.length-1)
    .map(x => XY(gw.take(x+1), gb.take(x+1)) setName "Adam")
    .toList

val animatedChart = 
     Chart(animated = true, transition_duration=simpleGw.length, animate_multiple_series = true)
     //.addSeries(contour)
     .addSeries(simpleGDAnimation)
     .addSeries(adamAnimation)
     .setTitle("Gradient Trace")
     .setLegend(x = 0.5, y = -0.5, orientation = HORIZONTAL, xanchor = AUTO, yanchor = AUTO)
     .addAxes(Axis(X, title = "w"), Axis(Y, title = "b"), Axis(Z, title = "loss"))

animatedChart.plotInline

val countourChart = Chart()
     .addSeries(contour, adamGradient, simpleGDGradient)
     .setTitle("Loss Contour")
     .setLegend(x = 0.5, y = -0.5, orientation = HORIZONTAL, xanchor = AUTO, yanchor = AUTO)
     .addAxes(Axis(X, title = "w"), Axis(Y, title = "b"), Axis(Z, title = "loss"))

countourChart.plotInline
