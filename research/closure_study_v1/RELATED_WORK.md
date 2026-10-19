# Related work and novelty boundary: closure study v1

Review date: 2026-10-19. Scope: primary papers and author manuscripts listed below. Relevant full-text sections and bibliographic records were inspected. This is a focused review, not an exhaustive search proving that a mechanism is absent from prior work.

## Research disposition

This successor is a **controlled representability, model-misspecification, and hierarchy diagnostic**. It does not establish architectural novelty, state-of-the-art performance, or a completed comparison with the methods reviewed here. Literature review is not an experimental reproduction of those methods.

The question is:

> For the declared open-system disc simulator, which errors follow from an action-only scene-momentum constraint and a mismatched observation-step update, and does hierarchy add value after those mismatches are corrected?

Explicit external effects, numerical integration, conservative internal interactions, equivariance, and graph hierarchies each have substantial prior art. In particular, external/internal momentum decomposition has a close precedent in MomentumGNN [10]; a broad claim of inventing that decomposition would be unsupported. A combination of established components requires evidence of a distinct mechanism or useful finding, not merely a new name.

The potentially useful contribution is the controlled diagnosis: establish a representational obstruction, isolate its correction from integration changes, and test whether an independent hierarchy benefit remains. Improvement over an intentionally constrained historical model is insufficient evidence of a new architecture or superiority to the field.

## Ten closest and foundational primary sources

The falsifiers below are proposals for this project. They are not claims that the cited authors ran these exact experiments.

### [1] Graph Network-based Simulators (GNS)

Sanchez-Gonzalez et al., *Learning to Simulate Complex Physics with Graph Networks*, ICML 2020. [Paper](https://proceedings.mlr.press/v119/sanchez-gonzalez20a/sanchez-gonzalez20a.pdf). BibTeX key: `sanchezgonzalez2020gns`.

**Inspected evidence:** Section 3 describes particle states with velocity history, material/boundary information, global context that can include external forces, and per-particle average-acceleration targets. Reported ablations identify message-passing depth and training noise as important for rollouts.

**Boundary:** Learning nodal acceleration with external context is established. The paper's trained simulator is not equivalent to every small graph baseline in this repository.

**Falsifier:** Give an adequately trained flat graph baseline the same information, integrator, data and tuning budget. If it explains the successor's gain, hierarchy has not earned scientific credit. Apply training-history and noise choices symmetrically.

### [2] MeshGraphNets

Pfaff et al., *Learning Mesh-Based Simulation with Graph Networks*, ICLR 2021. [Full text, v4](https://arxiv.org/html/2010.03409v4). BibTeX key: `pfaff2021meshgraphnets`.

**Inspected evidence:** Sections 3-4 describe mesh/world edges, node types distinguishing obstacles or boundaries, and task-specific output/update formulations. Cloth uses acceleration; other applications predict first-order changes or fields.

**Boundary:** Boundary representation and the state-update formulation are substantive model choices with established precedents. The paper covers mesh domains beyond the disc system studied here.

**Falsifier:** Supply boundary features and a consistent update to the flat baseline. If these changes explain the improvement, do not attribute it to the hierarchy. Keep contact-rich and contact-free errors separately visible.

### [3] MultiScale MeshGraphNets

Fortunato et al., *MultiScale MeshGraphNets*, 2022 manuscript. [Full text, v1](https://arxiv.org/html/2210.00612v1). BibTeX key: `fortunato2022multiscale`.

**Inspected evidence:** Sections 2 and 5 describe fine/coarse message passing with transfer operators and multigrid-like processing schedules. Coarse communication improves the reported accuracy/cost tradeoff. Coarse mesh construction matters, and accumulated rollout error remains unresolved.

**Boundary:** Coarse communication as a remedy for limited propagation is established motivation, not a RIPII novelty claim or a universal speed guarantee.

**Falsifier:** Compare hierarchy against spending the same measured compute on flat message passing. Include coarsening, transfers, dynamic graph construction and substeps in the cost. Reject an efficiency claim if accounting for those operations removes the advantage.

### [4] E(n) Equivariant Graph Neural Networks (EGNN)

Satorras, Hoogeboom and Welling, *E(n) Equivariant Graph Neural Networks*, ICML 2021. [Paper](https://proceedings.mlr.press/v139/satorras21a/satorras21a.pdf). BibTeX key: `satorras2021egnn`.

**Inspected evidence:** Section 3 constructs coordinate updates from invariant scalar messages and relative vectors. Section 3.2 adds a velocity-aware update with a learned velocity multiplier. The paper establishes Euclidean equivariance.

**Boundary:** Equivariance does not by itself require antisymmetric pair forces or fixed scene momentum. This is an inference from the update equations, not an additional theorem attributed to the authors.

**Falsifier:** Compare with a flat equivariant model receiving identical environmental information. Measure equivariance and force balance separately. Transform boundary geometry and external vectors along with particles in coordinate-transformation tests.

### [5] Equivariant Graph Hierarchy-Based Neural Networks (EGHN)

Han et al., *Equivariant Graph Hierarchy-Based Neural Networks*, NeurIPS 2022. [Full text, v2](https://arxiv.org/html/2202.10643v2). BibTeX key: `han2022eghn`.

**Inspected evidence:** Sections 3.1-3.3 define equivariant matrix message passing, learned soft pooling and equivariant unpooling. Low-level particles are grouped into a smaller system and higher-level information is returned to them. Theorem 2 concerns equivariance of these components.

**Boundary:** Learned clustering plus an equivariant graph hierarchy is direct prior art. The cited theorem is not a momentum-conservation theorem.

**Falsifier:** Compare geometric/learned grouping with fixed or identity grouping under the same surrounding model. For a field-level comparison, an appropriately adapted EGHN is a relevant competitor. An implementation adapted to this task must be identified as such.

### [6] Dissipative SymODEN

Zhong, Dey and Chakraborty, *Dissipative SymODEN: Encoding Hamiltonian Dynamics with Dissipation and Control into Deep Learning*, ICLR DeepDiffEq workshop 2020. [Paper](https://arxiv.org/pdf/2002.08860). BibTeX key: `zhong2020dissipativesymoden`.

**Inspected evidence:** Equation 3 includes energy, dissipation and external control. A factorized positive-semidefinite dissipation matrix gives a structural constraint. Training uses a differentiable ODE solver; the demonstrated tasks are low-dimensional mechanical systems.

**Boundary:** Combining physical structure with explicit dissipation/control is established. This is conceptual prior art, not a direct particle-contact benchmark equivalent.

**Falsifier:** Compare a flexible external-force network with a small constrained parametric closure, such as a learned nonnegative drag coefficient. If the latter explains the gain, retain that simpler explanation.

### [7] PhysGraph

Halimi et al., arXiv:2301.11841, 2023. [Full text, v2](https://arxiv.org/html/2301.11841v2). BibTeX key: `halimi2023physgraph`.

**Inspected evidence:** Sections 3.1-3.3 separate a force module, based on specified potentials, from a learned displacement/integration module. The inspected v2 focuses on iterative quasi-static cloth refinement initialized from coarse geometry.

**Boundary:** Force/integration modularity is prior art. Quasi-static refinement is not evidence of dynamic drag/contact rollout performance. The record title is *Physics-Based Integration Using Graph Neural Networks*; the v2 full-text title is *Physics-Based Cloth Enhancement Using Graph Neural Networks*.

**Falsifier:** When analytical external forces are supplied to RIPII, include an equally informed conventional numerical baseline. A gain obtained by supplying more accurate force information is not evidence of a superior learned mechanism.

### [8] Numerical integration and learned dynamics

Zhu et al., *On Numerical Integration in Neural Ordinary Differential Equations*, ICML 2022. [Paper](https://proceedings.mlr.press/v162/zhu22f/zhu22f.pdf). BibTeX key: `zhu2022integration`.

**Inspected evidence:** Inverse modified differential equations analyze how a training solver affects the learned vector field. The error analysis separates discretization and learning errors. The Hamiltonian results have explicit assumptions, including conditions on the system and integrator; the paper also states limitations of its analysis.

**Boundary:** Matching one discrete sampling map does not establish identification of the continuous physical law. The results are not blanket conservation guarantees for all learned simulators.

**Falsifier:** Freeze force parameters and vary step size/substeps. If accurate behavior only persists at the training discretization, scope the claim to a discrete surrogate. Record integration changes as scientific changes.

### [9] DYNAMI-CAL GraphNet

Sharma and Fink, *Dynami-CAL GraphNet: A Physics-Informed Graph Neural Network Conserving Linear and Angular Momentum for Dynamical Systems*, 2025 manuscript. [Full text, v2](https://arxiv.org/html/2501.07373v2). BibTeX key: `sharma2025dynamical`.

**Inspected evidence:** The manuscript combines edge-local equivariant internal interactions, reflected boundary nodes and a separate nodal external-velocity contribution. It distinguishes open dissipative dynamics from closed-system conservation and discusses complex unobserved external effects as a remaining limitation.

**Boundary:** Conservation, dissipation, boundaries and external effects already coexist in a closely related particle model. A related journal version exists; its full text was not inspected in this review.

**Falsifier:** Use a boundary-aware flat conservative interaction model as a serious rival. Reject a hierarchy claim supported only by comparisons against models deprived of boundary or external-force information.

### [10] MomentumGNN

Wang et al., *Momentum-Conserving Graph Neural Networks for Deformable Objects*, April 2026 preprint. [Full text, v1](https://arxiv.org/html/2604.26097v1). BibTeX key: `wang2026momentumgnn`.

**Inspected evidence:** Section 4.1, Equations 8-10, separates an external-force momentum step from learned momentum-conserving corrections. Edge decoders predict stretching/bending impulses; geometry is updated between layers. A separate velocity projection addresses angular momentum at the final positions.

**Boundary:** The external/internal decomposition is close prior art. Its deformable-mesh application is distinct from the disc simulator, and the work is a preprint.

**Falsifier:** Compare hierarchy against a flat external-step-plus-conservative-impulse model. If hierarchy contributes no reproducible benefit, retain the simpler model. Check physical constraints on the complete returned state, not an isolated force layer.

## Identifiability and equal information

The experiment must state which environment quantities are observed, fixed, estimated or withheld. If drag strength or wall geometry changes while the input state/action remains identical, the next state can differ. A deterministic one-step predictor cannot identify both outcomes from that identical input alone. This is an information limitation, not something a larger hierarchy can remove.

An environment-shift test must therefore either supply the varying parameters to every competitor or supply a declared history from which those quantities may be identified. The latter still requires an identifiability argument and an empirical check. The proposed model must not receive true coefficients, wall normals, simulator internals or latent labels that other learned baselines lack.

Separate two settings:

1. **Known physical closure:** Expose the same known forces/parameters to all relevant baselines. Include the conventional numerical model. The question concerns the remaining learned component or its cost.
2. **Identified closure:** Estimate coefficients only from the allowed training/development information, retain estimates and uncertainty where applicable, and freeze them before protected evaluation.

A coordinate-equivariance test must transform the complete physical problem, including wall geometry and external-force directions. Rotating only particles inside fixed walls generally creates a different physical problem.

## Integration and conservation caveats

The observation interval, held-action convention, number of simulator substeps, force evaluation order and position-update rule belong in the model contract. Changing them can change both the hypothesis class and the target relation [8].

For a single particle with constant acceleration $a$, $S$ semi-implicit substeps over an observation interval $\Delta t$ give

$$
v_{n+1}=v_n+a\Delta t,\qquad
q_{n+1}=q_n+\Delta t\,v_n+\frac{S+1}{2S}a\Delta t^2.
$$

This follows by summing the $S$ updated velocities. The coefficient is $5/8$ for four substeps and $1$ for one step. This elementary example establishes a discrete update mismatch; it does not prove that matching substeps produces better learned performance generally. Contact switches, clipping and state-dependent forces require separate analysis.

Internal conservation concerns the chosen system boundary. The observed particles may exchange momentum with walls and other external agents. Dissipation need not violate pairwise momentum exchange, and momentum conservation does not imply energy conservation. Any claimed constraint must be checked after masks, mass handling, pooling, prolongation, force accumulation and integration. A proof for one component does not automatically extend to the whole model.

## Discriminating experiment and stopping interpretation

A minimal factorial study separates closure and integration:

| External closure | Observation update | Question isolated |
| --- | --- | --- |
| Absent | Historical one-step update | Historical constrained foundation |
| Present | Historical one-step update | Contribution of closure |
| Absent | Consistent substeps | Contribution of integration |
| Present | Consistent substeps | Corrected foundation |

Then compare flat and hierarchical models on the corrected foundation. This is a proposed control structure; the executable protocol determines the actual authorized cells, seeds, budgets and endpoints. A cell not executed cannot support an empirical claim.

Fix data identities, losses, development allowance and failure rules before comparing models. Report all failures. Charge every model for actual integration stages, graph construction and hierarchy operations. Distinguish matched optimizer-update budgets from matched wall time or arithmetic budgets; none substitutes automatically for the others.

The strongest result against the hierarchy hypothesis is that a corrected flat or small parametric model explains the gain with equal information and lower or equal cost. That is a valid terminal result. A favorable correction result establishes only its declared scope. It does not demonstrate superiority to GNS, EGNN, EGHN, DYNAMI-CAL GraphNet or MomentumGNN unless appropriate implementations were actually compared under a defensible protocol.

Proofs may establish representability, algebraic constraints or numerical consistency under stated assumptions. They cannot guarantee optimizer success, out-of-distribution accuracy, a practical efficiency advantage or a positive research outcome. Historical negative results and frozen protocols remain separate evidence; this successor does not rewrite them.

## References and citation metadata

Numbered entries [1]-[10] above correspond to the keys in [references.bib](references.bib), in the same order. URLs identify the primary versions inspected. No quantitative results from these papers are transplanted into this study, and no reviewed method is described as reproduced here.
