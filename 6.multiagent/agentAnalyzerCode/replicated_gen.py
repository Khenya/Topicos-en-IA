from fastapi import FastAPI
from pydantic import BaseModel
from scipy.optimize import minimize
from scipy.integrate import quad
from scipy import stats
from typing import List

app = FastAPI()

class OptimizationRequest(BaseModel):
    initial_value: float

class IntegrationRequest(BaseModel):
    lower_limit: float
    upper_limit: float

class StatisticsRequest(BaseModel):
    data: List[float]

@app.post('/optimize/')
def optimize(request: OptimizationRequest):
    """
    Optimizes a quadratic objective function f(x) = x^2 + 5x + 10.
    Returns the optimal value starting from the provided initial value.
    """
    def objective(x):
        return x**2 + 5 * x + 10
    
    result = minimize(objective, request.initial_value)
    return {'optimal_value': result.x.tolist()}

@app.post('/integrate/')
def integrate(request: IntegrationRequest):
    """
    Calculates the area under the curve of f(x) = x^2 between the lower and upper limits.
    """
    def integrand(x):
        return x**2
    
    area, error = quad(integrand, request.lower_limit, request.upper_limit)
    return {'area_under_curve': area, 'error_estimate': error}

@app.post('/statistics/')
def statistics(request: StatisticsRequest):
    """
    Calculates the mean and variance of a given list of numbers.
    """
    mean = stats.tmean(request.data)
    variance = stats.tvar(request.data)
    return {'mean': mean, 'variance': variance}
