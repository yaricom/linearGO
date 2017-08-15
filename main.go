package main

import (
	"fmt"
	"os"
	"io/ioutil"
	"strings"
	"io"
	"github.com/yaricom/linearGO/solvers/gd"
)

func initArray(x[][]float32, n int, insertIntercept bool) {
	m := len(x)
	if insertIntercept { n += 1 }
	for i := 0; i < m; i++ {
		x[i] = make([]float32, n)
		if insertIntercept {
			x[i][0] = 1 // insert intercept for first parameter
		}
	}
}

func parseInput(r io.Reader) (x [][]float32, x_test [][]float32, y []float32) {
	var n, m int
	fmt.Fscanf(r, "%d %d\n", &n, &m)
	x = make([][]float32, m)
	y = make([]float32, m)
	initArray(x, n, true)
	for i := 0; i < m; i++ {
		for j := 1; j < n + 1; j++ {
			fmt.Fscanf(r, "%f ", &x[i][j])
		}
		fmt.Fscanf(r, "%f ", &y[i])
	}
	fmt.Fscanf(r, "%d\n", &m)
	x_test = make([][]float32, m)
	initArray(x_test, n, true)
	for i := 0; i < m; i++ {
		for j := 1; j < n + 1; j++ {
			fmt.Fscanf(r, "%f ", &x_test[i][j])
		}
	}
	return x, x_test, y
}

func main() {
	if len(os.Args) < 2 {
		printHelp()
		os.Exit(0)
	}
	// read samples file
	sBytes, err := ioutil.ReadFile(os.Args[1])
	if err != nil {
		fmt.Print(err)
		os.Exit(1)
	}
	samplesStr := strings.TrimSpace(string(sBytes))

	x, x_test, y := parseInput(strings.NewReader(samplesStr))

	if len(x) != len(y) {
		fmt.Printf("Wrong size of parsed input data. x: " +
			"%d, y: %d, x_test: %d\n", len(x), len(y), len(x_test))
		os.Exit(1)
	}

	fmt.Println(x)
	fmt.Println(y)
	fmt.Println(x_test)

	// do model learning
	alpha, iters, paramConvDiff := float32(0.1), 10000, float32(10e-8)
	theta, err := gd.Learn(x, y, alpha, iters, paramConvDiff)
	if err != nil {
		fmt.Println(err)
	}
	fmt.Println("Learned parameters:")
	fmt.Println(theta)

	// do prediction
	y_test := gd.Predict(x_test, theta)
	fmt.Println("Predicted:")
	fmt.Println(y_test)
}

func printHelp()  {
	fmt.Println("Arguments:")
	fmt.Println("samples - the path to the file with data samples")
	fmt.Println("solver - the solver to use [Default: gd]")
}