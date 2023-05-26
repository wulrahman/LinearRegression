<?php
class Layer_slice {
    public $input_size, $weights, $bias, $learning_rate, $cost;
    function __construct($input_size, $learning_rate) {
        $this->input_size = $input_size;
        $this->weights = array_map(function () {
            return (rand(0, 1000)/1000);
        }, array_fill(0, $input_size, null));

        $this->bias = (rand(0, 1000)/1000);
        $this->learning_rate =  $learning_rate;
        $this->cost = 0;
    }

    function feedForward($inputs) {
        $sum = $this->bias;
        foreach($inputs as $key => $input) {
            $sum += $input * $this->weights[$key];
        }
        return $sum;
    }

    function backpropogation($inputs, $expcted_output) {
        $output = $this->feedForward($inputs);
        $error =  ($output - $expcted_output);
        $this->cost += pow($error,2);
        $this->bias = $this->bias - $this->learning_rate * ($error);
        foreach($inputs as $key => $input) {
            $this->weights[$key] = $this->weights[$key] - $this->learning_rate * $input * ($error);
        }
    }
}

class Layer {
    public $splices, $learning_rate, $input_size, $output_size;
    function __construct($input_size, $output_size, $learning_rate) {
        $this->learning_rate =  $learning_rate;
        $this->input_size = $input_size;
        $this->output_size = $output_size;
        $this->splices = array_map(function() {
            return new Layer_slice($this->input_size, $this->learning_rate);
        }, array_fill(0, $output_size, null));
    }

    function predict($input) {
        $output = array_fill(0, $this->input_size, null);
        foreach($this->splices as $key => $splice) {
            $output[$key] = $splice->feedForward($input);
        }
        return $output;
    }
    
    function backpropogation($input, $outputs) {
        foreach($this->splices as $key => $splice) {
            $splice->backpropogation($input, $outputs[$key]);
        }
    }
}

$network = new Layer(1, 10, 0.01);
for ($i = 0; $i < 1000; $i++) {
    $network->backpropogation([1], [1, 2, 3, 4, 5, 6, 7, 8 ,9 ,10]);
    $network->backpropogation([2], [0, 2, 3, 4, 5, 6, 7, 8 ,9 ,10]);
    $network->backpropogation([3], [0, 0, 3, 4, 5, 6, 7, 8 ,9 ,10]);
    $network->backpropogation([4], [0, 0, 0, 4, 5, 6, 7, 8 ,9 ,10]);
}
print_r($network->predict([5]));

$network2 = new Layer(2, 2, 0.01);
for ($i = 0; $i < 1000; $i++) {
    $network2->backpropogation([1, 1], [1, 1]);
    $network2->backpropogation([0, 1], [0, 0]);
    $network2->backpropogation([1, 0], [0, 0]);
    $network2->backpropogation([0, 0], [0, 0]);
}
print_r($network2->predict([1, 1]));
