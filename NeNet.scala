/*
Author: Charlie R. Hicks
Purpose: Neural Network to fit unknown functions with variable number of inputs and outputs.

 */

package NeNet {

  import scala.io.Source
  import scala.util.Random
  import scala.collection.mutable.ArrayBuffer
  import scala.compat.Platform
  import Math.exp

  object Util{
    //gives a random between the range of -0.1 and 0.1
    def randomDouble() = {      
      val random = (Random.nextDouble()/5)-.1
    }
    //This will be a function ran after the input layer to give a 0 or 1 based on input
    def sigmoid(x:Double):Double = {
      1/(1 + Math.exp(-x))
    }
    // assuming input is sigmoid output
    def sigmoidPrime(x:Double):Double = {
      x * (1 -x )
    }

  }

  //Need the number of inputs from the previous layer or number of incoming inputs in the case of an input node
  class Node(numInputs:Int){
    //+1 for bias
    //for number of weights incoming
    var weights = Array.fill(numInputs+1)(Util.randomDouble())
    var output = 0.0
    var error : Double = 0.0
    def calculate(inputs: Array[Double]) : Double = {
      //weight would be multiplied by 1 for bias
      var sum = weights(0)
      for(i <- 0 until inputs.size){
        //again bais is 0 in weights.
        sum += inputs(i) * weights(i+1)
      }

      //output = Util.sigmoid(sum)
      //output
      Util.sigmoid(sum)
    }


/*    def Loudcalculate(inputs: Array[Double]) : Double = {
      //weight would be multiplied by 1 for bias
      var sum = weights(0)
      for(i <- 0 until inputs.size){
        //again bais is 0 in weights.
        sum += inputs(i) * weights(i+1)
      }

      output = Util.sigmoid(sum)
      output
 }*/
    
    def sumWeights() = {
      weights.sum
    }
    def runError(delta:Array[Double]) = {
      var sum = 0.0

    }
    def updateWeights(activations:Array[Double],alpha:Double, error:Double){
    weights(0) = weights(0) + alpha * error
      for(i <- 1 until weights.length){
      weights(i) = weights(i) + alpha * activations(i-1) * error
      }
    }
    override def toString() = {
    var x = ""
      for( i <- weights.indices){
        x += "  Weight" + (i+1) + " " + weights(i).toString + "\n"
      }
      x
    }
  }
  class Layer(numberOfNodes:Int,numberOfInputs:Int){

    val nodes =  Array.fill(numberOfNodes)(new Node(numberOfInputs))
    val errors = Array.fill(numberOfNodes)(0.0)
    //for input layer
    def setOutputs(answers:Array[Double]) = {
      for(i<- answers.indices){
        nodes(i).output = answers(i)
      }

    }
    def updateWeights(activations:Array[Double], alpha:Double) = {
    for( i <- nodes.indices){
      nodes(i).updateWeights(activations,alpha,errors(i))
    }
  }
    def getIthWeightsSum(i:Int): Double ={
      var sum = 0.0
      for(j <- nodes.indices){
        //i+1 because bias
        sum += nodes(j).weights(i+1) * errors(j)
      }
      sum
    }


    override def toString() : String = {
      var x = ""
      for(i <- nodes.indices){
        x += " Node" + (i+1) + "\n" + nodes(i).toString + "\n"
      }
      x
    }
    def getOutputs():Array[Double] = {
      nodes.map( (x) => {x.output})
    }


  }


  class NeuralNet(numberOfInput:Int,numberOfOutput:Int, var layersConstruct:Array[Int]){
    //output layer and input layer
    val layers = new Array[Layer](layersConstruct.length+2)

    //prepend input layer and append output layer
    layersConstruct = (numberOfInput +: layersConstruct :+ numberOfOutput)
    for( i <- 0 until layersConstruct.length){
      if(i == 0){
        layers(0) = new Layer(layersConstruct(i),numberOfInput)
      }
      else{
        layers(i) = new Layer(layersConstruct(i),layersConstruct(i-1))
      }
    }

    def clean(x:Int) : Unit = {
      for(l <- layers.indices){
        for(i <- layers(l).nodes.indices){
          for(j <- layers(l).nodes(i).weights.indices){
            layers(l).nodes(i).weights(j) = Util.randomDouble()
          }
        }
      }
    }

    def run(input:Array[Double],scalingFactor:Double) : Array[Double]  = {
      //divide by scaling maybe
      val correctInputs = input.map( (x) => { x})

      layers(0).setOutputs(correctInputs)
      for(l <- 1 until layers.length){
        //get activations
        var a = layers(l-1).getOutputs()
        var nodes = layers(l).nodes
        //now run for the rest of the layers
        for( i<- nodes.indices){
          nodes(i).calculate(a)
        }
      }
        layers(layers.length -1).getOutputs.map( (x) => { x * scalingFactor })

    }
    override def toString() : String = {
      var x = ""
      for(i <- 0 until layers.length){
        x += "Layer" + (i+1) + "\n" + layers(i).toString + "\n"

      }
      x
    }

    def backPropLearning(inputs:Array[Array[Double]],outputs:Array[Array[Double]],alpha:Double,time:Double,scalingFactor:Double) = {
      val end = time * 60 * 1000 + (Platform.currentTime)
      while(Platform.currentTime   < end){

        for(e <- inputs.indices){

          //propagate the inputs forward to compute the outputs
          //divide by scaling maybe
        val scaledInputs = inputs(e).map( (x) => { x })
          layers(0).setOutputs(scaledInputs)


          for(l <- 1 until layers.length){
            //get activations
          var a = layers(l-1).getOutputs()
            var nodes = layers(l).nodes
            //now run for the rest of the layers
            for( i<- layers(l).nodes.indices){
              nodes(i).calculate(a)
            }

          }
          //get output layer
          var layer = layers(layers.length-1)
          for(j <- layer.nodes.indices){
            layer.errors(j) = Util.sigmoidPrime(layer.nodes(j).output) * ((outputs(e)(j)/scalingFactor) - layer.nodes(j).output)
          }

          for(l <- (layers.length - 2) to 1 by -1){
            layer = layers(l)
            for(i <- layer.nodes.indices){
              layer.errors(i) = Util.sigmoidPrime(layer.nodes(i).output) * layers(l+1).getIthWeightsSum(i)
            }
          }
          for(l <- 1 until layers.length){
            layer = layers(l)
            layer.updateWeights(layers(l-1).getOutputs,alpha)

          }

        }

      }
    }

  }
}
