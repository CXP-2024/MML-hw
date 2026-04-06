# Problem Set 2 Reference Answers

## 1. Denoising Diffusion Probabilistic Models

### 1.1 DDPM Implementation

The three functions that students need to implement in the `DDPM` class:

#### `forward_process(self, x0, t)`

The forward process samples \( x_t \) from \( q(x_t \mid x_0) = \mathcal{N}(\sqrt{\bar\alpha_t}\, x_0,\, (1-\bar\alpha_t)\,I) \):

```python
x_t = torch.sqrt(alpha_bars) * x0 + torch.sqrt(1 - alpha_bars) * epsilon
return x_t, epsilon
```

#### `loss_fn(self, x0)`

The denoising loss is the MSE between the predicted noise and the actual noise:

```python
epsilon_pred = self.backbone(x_t, t)
loss = torch.mean((epsilon_pred - epsilon) ** 2)
return loss
```

#### `reverse_process(self, xt)` (loop body)

The DDPM stochastic reverse step:

```python
epsilon_pred = self.backbone(xt, time)
sigma = torch.sqrt((1 - self.alpha_prev_bars[t]) * self.betas[t] / (1 - self.alpha_bars[t]))
xt = (1 / torch.sqrt(self.alphas[t])) * (xt - (self.betas[t] / torch.sqrt(1 - self.alpha_bars[t])) * epsilon_pred) + sigma * noise
```

## 2. Conditional Flow Matching

### 2.1 Conditional Velocity Field

(a) Given \( x_t = (1-t)\,x_0 + t\,x_1 \):

\[
u_t(x_t \mid x_1) = \frac{dx_t}{dt} = -x_0 + x_1 = x_1 - x_0
\]

(b) The CFM training objective:

\[
\mathcal{L}_{\text{CFM}}(\theta) = \mathbb{E}_{t \sim \mathcal{U}[0,1],\, x_0 \sim \mathcal{N}(0,I),\, x_1 \sim q}\bigl[\| v_\theta(x_t, t) - (x_1 - x_0) \|^2\bigr]
\]

This is easier to compute than the marginal FM loss because:
- The marginal FM loss requires computing the marginal velocity \( u_t(x) = \mathbb{E}[u_t(x \mid x_1) \mid X_t = x] \), which involves an intractable posterior.
- The CFM loss uses the conditional velocity directly, requiring only samples from the data distribution and the noise distribution.
- Lipman et al. proved that the gradients of the CFM loss and the marginal FM loss are equal, so minimizing CFM yields the same optimal model.

### 2.2 CFM Loss Implementation

```python
def cfm_loss(model, x1, device):
    batch_size = x1.size(0)
    x0 = torch.randn_like(x1).to(device)
    t = torch.rand(batch_size, device=device)
    t_expand = t.unsqueeze(-1)
    x_t = (1 - t_expand) * x0 + t_expand * x1
    target = x1 - x0
    pred = model(x_t, t)
    loss = torch.mean((pred - target) ** 2)
    return loss
```

### 2.3 Euler ODE Solver Implementation

```python
def euler_ode_solve(model, x0, n_steps=100, device=None):
    if device is None:
        device = x0.device
    dt = 1.0 / n_steps
    x = x0.clone()
    trajectory = [x.clone()]
    for i in range(n_steps):
        t = torch.ones(x.size(0), device=device) * (i * dt)
        v = model(x, t)
        x = x + v * dt
        trajectory.append(x.clone())
    return x, trajectory
```

### 2.4 Analysis

1. **Step count comparison:**
   - 1 step: Very poor quality (points are spread randomly, not concentrated at modes).
   - 5 steps: Some mode structure visible but blurry.
   - 10 steps: Reasonable quality, 8 clusters visible.
   - 50-100 steps: High quality, clean 8-Gaussian clusters.
   - The velocity field learned by CFM needs sufficient integration steps because the marginal velocity field is nonlinear.

2. **Velocity field evolution:**
   - At t=0: The velocity field points from the noise distribution toward the data modes. Vectors are relatively uniform in direction.
   - At t=0.25-0.5: The field becomes more structured, with clear convergence toward the 8 modes.
   - At t=0.75: The velocity field shows fine refinement around each mode.
   - The field transitions from global transport to local refinement.

## 3. Mean Flow

### 3.1 MeanFlow Identity

**(a) Derivation:**

Start from the definition of the ODE trajectory:
\[
z_t = z_0 + \int_0^t v(z_s, s)\,ds
\]

So \( z_t - z_0 = \int_0^t v(z_s, s)\,ds \), and the mean velocity is:
\[
\bar{u}(z_t, t) = \frac{z_t - z_0}{t} = \frac{1}{t}\int_0^t v(z_s, s)\,ds
\]

Now differentiate \( \bar{u} \) with respect to \( t \) along the ODE trajectory. Using the product rule on \( (z_t - z_0) = t \cdot \bar{u} \):

\[
\frac{d}{dt}[t \cdot \bar{u}] = v(z_t, t)
\]

Expanding:
\[
\bar{u} + t \cdot \frac{d\bar{u}}{dt} = v(z_t, t)
\]

Rearranging:
\[
\bar{u}(z_t, t) = v(z_t, t) - t \cdot \frac{d\bar{u}(z_t, t)}{dt}
\]

This is the **MeanFlow Identity**.

**(b) Limit as t → 0:**

Using L'Hôpital's rule:
\[
\lim_{t \to 0^+} \bar{u}(z_t, t) = \lim_{t \to 0^+} \frac{z_t - z_0}{t} = \lim_{t \to 0^+} \frac{dz_t/dt}{1} = v(z_0, 0)
\]

So the mean velocity at the start equals the instantaneous velocity.

**(c) At t = 1:**

\[
\bar{u}(z_1, 1) = \frac{z_1 - z_0}{1} = z_1 - z_0
\]

This means: if we can predict \( \bar{u}(z_1, 1) \) from knowing the trajectory, we can compute the full displacement in one step. However, this is not directly usable because \( \bar{u} \) at \( t=1 \) depends on \( z_1 \) (the endpoint), which we don't know until we solve the ODE. A practical approach is to train a displacement network \( D_\theta(z_0) \) that directly predicts \( z_1 - z_0 \) given only the starting noise \( z_0 \).

### 3.2 Numerical Verification

```python
def compute_meanflow_data(model, n_samples, n_steps=100, device=None):
    if device is None:
        device = next(model.parameters()).device
    z0 = torch.randn(n_samples, 2).to(device)
    dt = 1.0 / n_steps
    z = z0.clone()
    trajectory = [z.clone()]
    times = [0.0]
    for i in range(n_steps):
        t_val = i * dt
        t = torch.ones(z.size(0), device=device) * t_val
        v = model(z, t)
        z = z + v * dt
        trajectory.append(z.clone())
        times.append((i + 1) * dt)
    return z0, trajectory, times
```

Students should then:
1. Compute \( \bar{u} = (z_t - z_0)/t \) for each \( t > 0 \)
2. Compute \( v(z_t, t) \) from the model
3. Compute \( d\bar{u}/dt \) numerically (finite differences)
4. Verify \( \bar{u} \approx v - t \cdot d\bar{u}/dt \)

### 3.3 One-Step Generation

The displacement network is trained on trajectory pairs:
- Input: \( z_0 \) (noise)
- Target: \( z_1 - z_0 \) (displacement)

```python
# Training data
z0_all, traj, times = compute_meanflow_data(velocity_model, n_samples=100000, n_steps=100)
z1_all = traj[-1]
displacement_target = z1_all - z0_all

# One-step generation
z0_new = torch.randn(10000, 2).to(device)
with torch.no_grad():
    z1_pred = z0_new + displacement_model(z0_new)
```

**Expected results:**
- 100-step ODE: Clean 8-Gaussian clusters, high quality.
- 1-step mean flow: The 8 modes should be visible but with more spread/blur compared to 100-step ODE. The network learns an average mapping from noise to data, which is inherently limited because the mapping is multi-modal (one noise sample could map to any of 8 modes).
- The quality gap illustrates why multi-step generation produces better results, and motivates more sophisticated mean flow training (e.g., using the MeanFlow Identity as a self-consistency loss, as in Geng et al., 2025).
