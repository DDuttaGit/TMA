# On the Existence of Universal Simulators of Attention

### _Anonymous Submission_
<p align="center">
<img src="https://github.com/user-attachments/assets/1ed3d235-7528-4f08-98e2-7a248f89bf77" />
</p>

Construction of the transformer network $\mathcal{U}$ (see the diagram above) has been algorithmically achieved using Restricted Access Sequence Programming [^1]. Realization of such a conceptual representation of $\mathcal{U}$ can be implemented using `Tracr` [^2].

Implementation of fundamental matrix operations and activations has been attached in this repository.

To interpret the diagram in the file `Inverse.pdf`, we request the readers to go top-down (layer 0 - 5). Inverting a matrix of order $3$, such as the following has been presented in the figure.

$$
\left(\begin{array}{ccc} 
7 & 8 & 12\\
10 & 11 & 9\\
2 & 4 & 21
\end{array}\right)
$$ 


#### References
[^1]: Weiss, Gail, Yoav Goldberg, and Eran Yahav. "Thinking Like Transformers." In International Conference on Machine Learning, pp. 11080-11090. PMLR, 2021.
[^2]: Lindner, David, János Kramár, Sebastian Farquhar, Matthew Rahtz, Tom McGrath, and Vladimir Mikulik. "Tracr: Compiled transformers as a laboratory for interpretability." Advances in Neural Information Processing Systems 36 (2023): 37876-37899.
