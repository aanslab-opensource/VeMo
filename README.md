## Introduction to VeMo (_Vehicle Modeler_)     

Developing a dynamic model for a high-performance
vehicle is a complex problem that requires extensive structural
information about the system under analysis. This information
is often unavailable to those who did not design the vehicle.
This paper proposes a lightweight encoder-decoder model based
on gate recurrent unit (GRU) layers to correlate the vehicle’s
future state with its past states, measured onboard, and control
actions the driver performs. The results demonstrate that the
model achieves a maximum mean relative error below 3%. It
also shows good robustness when subject to noisy input data.
Furthermore, being entirely data-driven and free from physical
constraints, the model exhibits good physical consistency in the
output signals, such as longitudinal and lateral accelerations, yaw
rate, and the vehicle’s longitudinal velocity.

The problem addressed in this study can be mathematically described using a uniform time-step discretization tₙ = n Δt, where n = 0, 1, 2, ...  is the subscript addressing the time index and Δt is the (constant) time step. The value of Δt is defined by the sampling rate of the sensors, set to 100 Hz in this work.

The problem is stated as searching for a neural network model F that computes the new state xₙ₊₁ as a function of the immediately past k states and control actions, i.e.,
  
xₙ₊₁ = F(xₙ₋ₖ₊₁, ..., xₙ; uₙ₋ₖ₊₁, ..., uₙ),  


where xⱼ ∈ ℝ⁴ is the vehicle state vector composed of the longitudinal acceleration aₓ, the lateral acceleration aᵧ, the yaw rate θ̇ , and the longitudinal velocity vₓ, i.e.,

x = [aₓ, aᵧ, θ̇, vₓ]ᵀ,


whereas uⱼ ∈ ℝ⁴ is the control action vector defined as

u = [uₜ, uᵦ, uₛ, u₉]ᵀ,


where uₜ, uᵦ, uₛ, and u₉ are the throttle percentage, the brake percentage, the steering angle, and the gear, respectively. Both past vehicle states and control actions (i.e., xⱼ and uⱼ with j ≤ n) are considered to capture possible delays in the effect of controls on the vehicle state.









## Usage on Local Machine: 
__Fedora Linux Users__
```
sudo dnf install python3.10
git clone https://github.com/aanslab-opensource/VeMo
cd VeMo/
python3.10 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python3 ./main_training_inference.py
```
__Windows Users__
- Download and install Python 3.10 from [python.org](https://www.python.org/).  
  Ensure you check the option to "Add Python to PATH" during installation.
```
git clone https://github.com/aanslab-opensource/VeMo
cd VeMo
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python ./main_training_inference.py
```
## Requirements:
| **Dependency**    | **Version** |
|--------------------|-------------|
| Python            | 3.10.12     |
| scipy             | 1.11.4      |
| numpy             | 1.25.2      |
| pandas            | 2.0.3       |
| keras             | 2.15.0      |
| tensorflow        | 2.15.0      |
| matplotlib        | 3.7.1       |
| graphviz          | 0.20.3      |
| pydot             | 1.2.2       |

## Results:

![vemo_telemetry_comparison](https://github.com/user-attachments/assets/ff5e0ff3-927c-4709-b965-bb3a166bea9e)
[vemo_telemetry_comparison.pdf](https://github.com/user-attachments/files/18409173/vemo_telemetry_comparison.pdf)


