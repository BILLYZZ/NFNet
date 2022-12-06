# NFNet Test version

Some demonstrations and tutorials:
1. Demo_1_Model a continuous Hamiltonian
2. Demo_2_Compare_Runtimes
3. Demo_3_Multi_Layer_Circuit
4. Demo_4_Born_Machine
5. Demo_5_MaxCut
6. Demo_6_196_qubit_Pattern_Recognition

Using the non-interacting Fermionic Formalism to efficiently simulate the measurements on a quantum circuit in polynomial time of the number of qubits, without the nearest-neighbor constraint, NFNet is a classical simulation platform of quantum systems composed of these three Hamiltonians: IZ+ZI, XX+YY, and XY+YX.

The mapping between Fermionic and Pauli Hamiltonian evolutions is based on the Jordan-Wigner transformation:

$\begin{align} \label{JW}
&a_j^{\dagger}:=\left(\prod_{k=1}^{j-1} Z_k\right)\left(\frac{X_j-i Y_j}{2}\right) \\
&a_j:=\left(\prod_{k=1}^{j-1} Z_k\right)\left(\frac{X_j+i Y_j}{2}\right),
\end{align}$

where the "multiplication" operations are tensor products. The creation and annihilation operators follow the anti-commuting rules:

$\left\{a_{i}, a_{j}\right\} \equiv a_{i} a_{j}+a_{j} a_{i}=0, \quad\left\{a_{i}^{\dagger}, a_{j}^{\dagger}\right\}=0, \quad\left\{a_{i}, a_{j}^{\dagger}\right\}=\delta_{i j} I$

In this example we look at a Fermion-preserving Hamiltonian on two fermion modes $i$ and $j$, written as (assuming $i<j$):

$\begin{equation}
\mathcal{H}_{Fermi}=b_{i i} a_i^{\dagger} a_i+b_{j j} a_j^{\dagger} a_j+b_{i j} a_i^{\dagger} a_j+b_{i j}^* a_j^{\dagger} a_i.
\end{equation}$

Because $H$ is hermitian, $a_{ii}, a_{jj}$ are real. We express $b_{ij}$ as $c+di$, where $c$ and $d$ are real continuous numbers.

In the most general case, $H$ corresponds to a Hamiltonian (in Pauli matrix form) which describes the interaction between qubits $i$ and $j$:

$\begin{equation}
    \begin{aligned}
        \mathcal{H}_{Pauli} &=b_{i i} a_i^{\dagger} a_i+b_{j j} a_j^{\dagger} a_j+(c+di) a_i^{\dagger} a_j+(c-di) a_j^{\dagger} a_i \\ &= -\frac{1}{2} \left(b_{ii} Z_i + b_{jj} Z_j\right)+\frac{c}{2} \left(Y_i \left(\prod_{k=i+1}^{j-1} Z_k\right) Y_j + X_i \left(\prod_{k=i+1}^{j-1} Z_k\right) X_j \right)\\
        &+\frac{d}{2}\left(Y_i \left(\prod_{k=i+1}^{j-1} Z_k\right) X_j - X_i \left(\prod_{k=i+1}^{j-1} Z_k\right) Y_j \right),
    \end{aligned}
\end{equation}$

In this example we first simulate the Z-basis measurement probabilities on the final state $|\psi_{f} \rangle = e^{-i\mathcal{H} t} |\psi_0 \rangle$ after the time evolution of $e^{-i\mathcal{H} t}$ on an input product state $|\psi_0 \rangle$. This simulation is computed in <b>polynomial<b> time of the number of qubits $N$.

We then compare the measurement probabilities with the exact diagonalization calculation, keeping track of the full state. This scales <b>exponential<b> time of the number of qubits $N$.


<html lang="en">
<img src="https://latex.codecogs.com/gif.latex?s=\text { sensor reading }"/>
</html>
