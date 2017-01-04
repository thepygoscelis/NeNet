/*
Observations It always seems to pick an architecture with more nodes closer to the 
inputs and fewer closer to the outputs

For example for I choose 3 max layers and 4 max nodes
it chose 3 layers in the order 4 4 2 
 */

import NeNet._
import scala.io.Source
import scala.util.Random
import scala.collection.mutable.ArrayBuffer
import scala.compat.Platform
import Math.{exp,abs}
class Bin() {

  var inputData = new ArrayBuffer[Array[Double]]()
  var outputData = new ArrayBuffer[Array[Double]]()

  def appendInput(a:Array[Double]){
    inputData += a
  }
  def appendOutput(a:Array[Double]){
    outputData += a
  }
  def +(that:Bin) : Bin = {
    this.inputData = this.inputData ++ that.inputData
    this.outputData = this.outputData ++ that.outputData

    this
  }
}
object NeuralNetArchitectureDriver {

  def kfold(neuralNet:NeuralNet,k:Int,inputs : Array[Array[Double]], outputs : Array[Array[Double]],learningRate:Double,scalingFactor:Double,time:Double) = {
    var kbins = Array.fill(k)(new Bin ())
    val indexes = Random.shuffle(Array.range(0,inputs.length,1).toList)
    var x = 0
    var error = 0.012
    //put inputs and outputs into randomly in bins
    for(i <- 0 until indexes.length){
      kbins(x).appendInput(inputs(i))
      kbins(x).appendOutput(outputs(i))
      x = (x+1)%k
    }

    for(i <- kbins.indices){
      var validation = kbins(i)
      var trainingData = new Bin()
      for(j <- kbins.indices){
        if(j != i){
          trainingData + kbins(j)
        }
      }
      neuralNet.backPropLearning(trainingData.inputData.toArray,trainingData.outputData.toArray,learningRate,time,scalingFactor)


      val inputs = validation.inputData.toArray
      val realOutputs = validation.outputData.toArray
      for(k <- inputs.indices){
        val output = neuralNet.run(inputs(k),scalingFactor)
        for(i <- output.indices){
          error = error + abs(output(i) - realOutputs(k)(i))
        }
      }
      //reset for next training
      neuralNet.clean(1)
    }
    error/k


  }
  def combs(set: List[List[Int]]) : List[List[Int]] = {
    set match {

      case Nil => List(Nil)
      case head :: _ => head.flatMap(i => combs(set.tail).map(i :: _))
    }

  }
  def makePerms(maxNumOfLayers: Int, maxNumberOfNodes: Int) : Array[Array[Int]] = {
    var perms = new ArrayBuffer[Array[Int]]()
    for( i <- 1 to maxNumOfLayers){
      //I could make this one line, but I didn't because I want you to be able to read it.
        perms = perms ++  combs(List.fill(i)((1 to maxNumberOfNodes).toList)).toArray.map( x => x.toArray)
    }
    perms.toArray


  }
  def main(args:Array[String]) : Unit = {
    val inputs = Source.fromFile(args(0)).getLines
    val filename = inputs.next
    val k_bins = inputs.next.toInt
    val max_num_hidden_layers = inputs.next.toInt
    val max_num_hidden_nodes = inputs.next.toInt
    val learning_rate = inputs.next.toDouble
    val error_tolerance = inputs.next.toDouble



    val datafile = Source.fromFile(filename).getLines
    val scaleingFactor = datafile.next.toDouble
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

    val perms = makePerms(max_num_hidden_layers,max_num_hidden_nodes)
    var errors = new ArrayBuffer[Tuple2[Array[Int],Double]]()
    for(i<- perms.indices){
      var neuralNet = new NeuralNet(numberOfInputs,numberOfOutputs,perms(i))
      var error = kfold(neuralNet,k_bins, functionInputs, functionOutputs, learning_rate,scaleingFactor, error_tolerance)
      errors += Tuple2(perms(i),error)
      println("Architecture : ")
      for( k <- perms(i).indices){
        print(" Layer: " + (k+1) + " Nodes: " + perms(i)(k))
      }
      println("\n Error: " + error)
    }
    var highest = errors(0)
    for(i <- 1 until errors.length){
      if(errors(i)._2 == highest._2){
        if(errors(i)._1.sum < highest._1.sum){
          highest = errors(i)
        }
      }
      if(errors(i)._2 < highest._2){
        highest = errors(i)
      }
    }
    println("Best Architecture")
    val BestArchitecture = highest._1
    val BestError = highest._2
    for( i <- BestArchitecture.indices){
      print("Layer " + (i+1) + " Nodes: " + BestArchitecture(i))
    }
    println("\nError: " + BestError)
  }

}
