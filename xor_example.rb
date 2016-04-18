
# require File.dirname(__FILE__) + '/../../lib/ai4r/neural_network/backpropagation'
require "rubygems"
require "ai4r"
require 'benchmark'

times = Benchmark.measure do

    srand 1

    net = Ai4r::NeuralNetwork::Backpropagation.new([2, 2, 1])

    puts "Training the network, please wait."
    2001.times do |i|
      net.train([0,0], [0])
      net.train([0,1], [1])
      net.train([1,0], [1])
      error = net.train([1,1], [0])
      puts "Error after iteration #{i}:\t#{error}" if i%200 == 0
    end

    puts "Test data"
    puts "[0,0] = > #{net.eval([0,0]).inspect}"
    puts "[0,1] = > #{net.eval([0,1]).inspect}"
    puts "[1,0] = > #{net.eval([1,0]).inspect}"
    puts "[1,1] = > #{net.eval([1,1]).inspect}"
end

  puts "Elapsed time: #{times}"
