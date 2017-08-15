// The simple linear regression solver based on Gradient Descent
package gd

import (
	"errors"
	//"fmt"
)
// Do dependent variable predictions based on provided data samples (x) and learned model parameters (theta).
// Return the predicted dependent variable values per each data sample.
func Predict(x[][]float32, theta[]float32) []float32 {
	m, n := len(x), len(theta)
	y := make([]float32, m)
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			y[i] += x[i][j] * theta[j]
		}
	}
	return y
}

// Do gradient descent and model parameters update
func paramUpdate(x[][]float32, theta[]float32, y[]float32, alpha float32) {
	m, n := len(x), len(theta)
	ht := Predict(x, theta)
	//fmt.Println(ht)
	for j := 0; j < n; j++ {
		s := float32(0)
		for i := 0; i < m; i++ {
			s += (ht[i] - y[i]) * x[i][j]
		}
		// update parameters
		theta[j] = theta[j] - alpha * s / float32(m)
	}
	//fmt.Println(theta)
}

// Perform linear regression model model learning using provided data samples (x), ground truth values of dependent
// variable (y), and learning rate value (alpha). The maxIters will be used to limit number of iterations for model
// learning if learned model parameters fail to converge withing paramConvDiff range.
// Return the learned regression model parameters which can be used for predictions.
func Learn(x[][]float32, y[]float32, alpha float32, maxIters int, paramConvDiff float32) ([]float32, error) {
	theta := make([]float32, len(x[0]))
	theta_prev := make([]float32, len(theta))
	iter := maxIters
	converged := false
	for ; iter > 0; iter-- {
		paramUpdate(x, theta, y, alpha)
		theta_diff := float32(0)
		for j := 0; j < len(theta); j++ {
			theta_diff += (theta[j] - theta_prev[j]) * (theta[j] - theta_prev[j])
		}
		//fmt.Println(theta_diff)
		if theta_diff <= paramConvDiff {
			converged = true
			break // model parameters converged within defined range
		}
		copy(theta_prev, theta)
	}

	if converged == false {
		return theta, errors.New("Warning! The model parameters failed to converge!")
	}

	return theta, nil
}


