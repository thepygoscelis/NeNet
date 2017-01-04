import NeNet._
import scala.io.Source
import scala.util.Random
import scala.collection.mutable.ArrayBuffer
import scala.compat.Platform
import Math.{exp,abs}

object NeuralNetDriver {

  def main(args:Array[String]): Unit = {

    val inputs = Source.fromFile(args(0)).getLines
    val filename = inputs.next
    val hidden = inputs.next.toInt

    val nodes = new Array[Int](hidden)
    var i = 0
    for( i <- 0 to (nodes.length-1)){
      nodes(i) = inputs.next.toInt
    }
    val learningRate = inputs.next.toDouble
    val error = inputs.next.toDouble
    val datafile = Source.fromFile(filename).getLines
    val scaleingFactor = datafile.next.toDouble
    //problem?
    val functionArityAndOutput = datafile.next.split(",").map( (x) => {x.toInt})
    val numberOfInputs = functionArityAndOutput(0)
    val numberOfOutputs = functionArityAndOutput(1)


    var inputData = new ArrayBuffer[Array[Double]]()
    var outputData = new ArrayBuffer[Array[Double]]()
    var tmp = new Array[Double](numberOfInputs+numberOfOutputs)
    while(datafile.hasNext){
      tmp = datafile.next.split(",").map( (x) => {x.toDouble})
      inputData += tmp.take(numberOfInputs)
      outputData += tmp.takeRight(numberOfOutputs)
    }
    val functionInputs = inputData.toArray
    val functionOutputs = outputData.toArray

    var neural_net = new NeuralNet(numberOfInputs,numberOfOutputs,nodes)


    neural_net.backPropLearning(functionInputs,functionOutputs,learningRate,error,scaleingFactor)


    var sum = 0.0
    for(i <- functionInputs.indices){
      val output = neural_net.run(functionInputs(i),scaleingFactor)
      print("Inputs: ")
      functionInputs(i).foreach( x => print(x + " "))
      println()
      print("Outputs: ")      
      functionOutputs(i).foreach( x => print(x + " " ))
      println()
      print("Net Outputs: ")
      output.foreach( x => print (x + " "))
      println()
      for(j <- output.indices){

        sum += abs(output(j) - functionOutputs(i)(j))
      }
    }

    println(" Total Error: " + sum)
  }

}
