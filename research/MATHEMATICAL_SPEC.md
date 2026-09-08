# Mathematical specification

## Legacy structured autoencoder

For input (x\in\mathbb R^D), the stochastic encoder produces

\[
(\mu,\log\sigma^2)=E_\theta(x),\qquad z_0=\mu+\sigma\odot\epsilon,
\quad \epsilon\sim\mathcal N(0,I),
\]

with (z_0=\mu) in evaluation mode. Each projective level orthonormalizes a learned
basis (B_l\in\mathbb R^{d\times r}), forms (P_l(z)=zB_lB_l^\top), and applies

\[
z_{l+1}=\operatorname{LN}\{z_l+g_l(z_l)[P_l(z_l)+0.25R_l(z_l)+0.1M_l(z_l)]\}.
\]

The final latent is split into (N) nodes. Sparse learned top-k attention performs
message passing, followed by attentive pooling (p). A transformation descriptor
(a\in\mathbb R^4) yields (c=A(z_0,a)) when enabled. With
(u=[z_L,p,\mu,c]), structural state is

\[
s=G(u)\odot F(u)+(1-G(u))\odot C(c).
\]

Hierarchical residual VQ chooses coarse and fine nearest code vectors,
(q=e_{k_1}+e_{k_2}), using straight-through gradients. The decoder computes
(\hat x=D([q,p,z_L])). The optimized objective is an uncertainty-balanced weighted
sum of reconstruction, KL, view consistency, projection, graph, moment, action,
depth, and VQ terms. A zero coefficient contributes neither data term nor learned
uncertainty offset. This large objective is weakly identifiable: several terms can
reward correlated latent statistics without improving reconstruction.

The simplest approximation is `plain_ae`: the same stochastic encoder family and a
decoder, with no nodes, graph, projection, action, fusion gate, or VQ. The negative
pilots support preferring this simpler explanation.

## Object-state world model

At time (t), object (i) has
(s_{t,i}=(p_x,p_y,v_x,v_y,r,m)\), force (a_{t,i}\in\mathbb R^2), and mask
(m_i\in\{0,1\}). All learned variants encode (h_i=E([s_i,a_i])) and predict a
bounded correction (\delta_i=\tanh H(h_i')\). The shared transition is

\[
v'_{i}=v_i+0.5\delta_{i,v},\qquad
p'_{i}=p_i+\Delta t\,v'_i+0.05\delta_{i,p},
\]

while radius and mass are copied exactly. Graph interactions sum learned messages
over neighbors within distance 0.6. The direct global control appends the masked
scene mean. The multiscale variant assigns each object softly to (K) groups,

\[
A_{ik}=\operatorname{softmax}_k(W_ah_i),\quad
\bar h_k=\frac{\sum_i A_{ik}h_i}{\sum_i A_{ik}},
\]

performs all-to-all coarse interaction, unpools (A\bar h'), then refines locally.
This is permutation equivariant but not rotation/translation equivariant and it is
not adaptive computation.

For rollout horizon (T), with coordinate weights (w=(4,4,1,1)),

\[
L_t=\frac{\sum_{b,i,j}m_{bi}w_j(\hat s_{t,bij}-s_{t,bij})^2}
{\sum_{b,i}m_{bi}\sum_jw_j},\qquad
L=L_1+\tfrac12 T^{-1}\sum_{t=1}^T L_t+\lambda_qL_q.
\]

Selection minimizes validation position RMSE plus 0.25 velocity RMSE. Complexity of
graph and multiscale interaction is (O(BN^2h+BK^2h)) time and quadratic edge
storage; MLP is (O(BN h^2)), while Transformer attention is (O(BN^2h)).

## Scientific interpretation

Learned grouping encodes the assumption that coarse global interactions improve
long-horizon/OOD dynamics beyond local graph messages or ordinary global context.
The current five-seed evidence falsifies that advancement claim in this simulator:
the graph wins every aggregate OOD comparison. Active group assignments establish
gradient reachability and non-collapse, not causal utility.

## Long-range coupling intervention

The extension adds an optional, symmetric all-pairs harmonic force. For live objects
and coupling strength \(\kappa\geq0\),

\[
F^{\mathrm{global}}_i=-\frac{\kappa}{N}\sum_{j\ne i}(p_i-p_j).
\]

Every pair contributes equal and opposite force, so total momentum is conserved when
drag, walls, and external actions are absent. At \(\kappa=0\), the original simulator
is recovered exactly. This targeted counterexample makes nonlocal information useful
without giving a model future information. The decisive control is masked global
pooling: hierarchy must beat generic global context, not only a radius-limited graph.
