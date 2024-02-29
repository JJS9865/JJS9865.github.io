---
layout: post
title: Ellipse and Gaussian Distribution
tags: ML
color: brown
author: sylhare
categories: Example
excerpt_separator: <!--more-->
---

<head>
  <style>
    table.dataframe {
      white-space: normal;
      width: 100%;
      height: 240px;
      display: block;
      overflow: auto;
      font-family: Arial, sans-serif;
      font-size: 0.9rem;
      line-height: 20px;
      text-align: center;
      border: 0px !important;
    }

    table.dataframe th {
      text-align: center;
      font-weight: bold;
      padding: 8px;
    }

    table.dataframe td {
      text-align: center;
      padding: 8px;
    }

    table.dataframe tr:hover {
      background: #b8d1f3; 
    }

    .output_prompt {
      overflow: auto;
      font-size: 0.9rem;
      line-height: 1.45;
      border-radius: 0.3rem;
      -webkit-overflow-scrolling: touch;
      padding: 0.8rem;
      margin-top: 0;
      margin-bottom: 15px;
      font: 1rem Consolas, "Liberation Mono", Menlo, Courier, monospace;
      color: $code-text-color;
      border: solid 1px $border-color;
      border-radius: 0.3rem;
      word-break: normal;
      white-space: pre;
    }

  .dataframe tbody tr th:only-of-type {
      vertical-align: middle;
  }

  .dataframe tbody tr th {
      vertical-align: top;
  }

  .dataframe thead th {
      text-align: center !important;
      padding: 8px;
  }

  .page__content p {
      margin: 0 0 0px !important;
  }

  .page__content p > strong {
    font-size: 0.8rem !important;
  }

  </style>
</head>


## 1. Coordinates

#### 1.1 Coordinates with basis

- basis $\{\hat{x}_1 \;\hat{x}_2\}$ or basis $\{\hat{y}_1\; \hat{y}_2\}$



<center><img src = "image/1.png" width = "30%" height = "100%"></center>



$$\begin{aligned}

\vec{r}_X &= \begin{bmatrix}x_1\\x_2\end{bmatrix} \text{: coordinate of} \; \vec r \text{ in basis} \; \{\hat x_1 \;\hat x_2 \}  \\

\vec{r}_Y &= \begin{bmatrix}y_1\\y_2\end{bmatrix} \text{: coordinate of} \;\vec r \text{ in basis} \;\{\hat y_1 \;\hat y_2 \} \\ \\

\end{aligned} \\ \hat{x}_1\perp\hat{x}_2,\;\; \hat{y}_1\perp\hat{y}_2 \\ \left \| \hat{x}_1 \right \| = \left \| \hat{x}_2 \right \| =\left \| \hat{y}_1 \right \| =\left \| \hat{y}_2 \right \| = 1$$


#### 1.2 Coordinate transformation



$$ 

\begin{aligned}

&\triangleright\,1. \;\;\hat{y}_1 = \begin{bmatrix}1\\0\end{bmatrix}, \; \hat{y}_2 = \begin{bmatrix}0\\1\end{bmatrix} \;\;\;\Longrightarrow \;\;\; \begin{bmatrix}\hat{y}_1&\hat{y}_2\end{bmatrix} = \begin{bmatrix}1&0 \\0&1\end{bmatrix} = I \\\\ 



&\triangleright\,2. \;\; U^\top U = \begin{bmatrix}\hat{x}_1^\top \\ \hat{x}_2^\top \end{bmatrix} \begin{bmatrix}\hat{x}_1 & \hat{x}_2 \end{bmatrix} = \begin{bmatrix}\hat{x}_1^\top\hat{x}_1 & \hat{x}_1^\top\hat{x}_2 \\ \hat{x}_2^\top\hat{x}_1 & \hat{x}_2^\top\hat{x}_2 \end{bmatrix} = \begin{bmatrix}1 & 0 \\ 0 & 1 \end{bmatrix} \\\\ &\;\;\;\;\;\;\;\;\Longrightarrow \; U^\top = U^{-1}

\\\\

&\triangleright\,3. \;\;  \vec r   = x_1\hat{x}_1 + x_2\hat{x}_2  = y_1\hat{y}_1 + y_2\hat{y}_2

\\ &\;\;\;\;\;\,\;\;\;\;\;= \begin{bmatrix}\hat{x}_1&\hat{x}_2\end{bmatrix}\begin{bmatrix}x_1\\x_2\end{bmatrix} =\begin{bmatrix}\hat{y}_1&\hat{y}_2\end{bmatrix}\begin{bmatrix}y_1\\y_2\end{bmatrix} = \begin{bmatrix}y_1\\y_2\end{bmatrix}

\\\\ &\;\;\;\;\;\,\;\;\;\;\;=

U\begin{bmatrix}x_1\\x_2\end{bmatrix} = \begin{bmatrix}y_1\\y_2\end{bmatrix} \quad \left( U = \begin{bmatrix}\hat{x}_1 & \hat{x}_2\end{bmatrix}\right)

\\\\

&\;\;\;\;\;\,\;\;\;\;\;\Longrightarrow \begin{bmatrix}x_1\\x_2\end{bmatrix} = U^{-1}\begin{bmatrix}y_1\\y_2\end{bmatrix}= U^{T}\begin{bmatrix}y_1\\y_2\end{bmatrix}

\end{aligned}

$$



#### 1.3 Coordinate change

- $U$를 이용해 좌표변환을 할 수 있다.



$$\begin{align*}

\begin{bmatrix}x_1\\x_2\end{bmatrix} \quad &\underrightarrow{U} \quad \begin{bmatrix}y_1\\y_2\end{bmatrix} \\\\

\begin{bmatrix}y_1\\y_2\end{bmatrix} \quad &\underrightarrow{U^T} \quad \begin{bmatrix}x_1\\x_2\end{bmatrix} 

\end{align*}$$


- - -


## 2. Equation of an Ellipse

#### 2.1 Unit circle



<center><img src = "image/2.png" width = "15%" height = "100%"></center>



$$x_1^2 + x_2^2 = 1 \;\implies \;\begin{bmatrix}x_1 & x_2 \end{bmatrix}

\begin{bmatrix}1 & 0 \\ 

               0 & 1 \end{bmatrix}

\begin{bmatrix}x_1 \\

               x_2 \end{bmatrix} = X^\top I X = 1 $$


#### 2.2 Independent ellipse

<center><img src = "image/3.png" width = "12%" height = "20%"></center>



$$\begin{align*} 

\frac{x_1^2}{a^2} + \frac{x_2^2}{b^2} = 1 &\implies

\begin{bmatrix}x_1 & x_2 \end{bmatrix}

\begin{bmatrix}\frac{1}{a^2} & 0 \\ 

                           0 & \frac{1}{b^2} \end{bmatrix}

\begin{bmatrix}x_1 \\

               x_2 \end{bmatrix} = 1 \\

& \implies \begin{bmatrix}x_1 & x_2 \end{bmatrix}

\Sigma_x^{-1}

\begin{bmatrix}x_1 \\

               x_2 \end{bmatrix} = X^\top\Sigma_x^{-1}X = 1\\ \\                       

&\;\,\text{where. }  

\Sigma_x^{-1} =\begin{bmatrix}\frac{1}{a^2} & 0\\

                           0 & \frac{1}{b^2} \end{bmatrix}, \,\Sigma_x = \begin{bmatrix}a^2 & 0\\

                 0 & b^2 \end{bmatrix} 

\end{align*}$$



- $\Sigma_x$에 장축과 단축에 대한 정보가 담겨있다.


#### 2.3 Dependent ellipse (Rotated ellipse)

<center><img src = "image/4.png" width = "35%" height = "50%"></center>



- Basis $\hat{x}$ 에서는 independent

- Basis $\hat{y}$ 에서는 dependent



$$

\begin{matrix}\begin{bmatrix}x_1 \\ x_2 \end{bmatrix} = U^\top 

\begin{bmatrix}y_1 \\ y_2 \end{bmatrix}\end{matrix},\quad

\begin{matrix} \begin{aligned} &x= U^\top y \\ &Ux=y

\end{aligned} \end{matrix}

$$



- Now we know in basis $\{\hat{x}_1 \;\hat{x}_2\}$

$$

x^\top  \Sigma_x^{-1}x = 1 \quad \text{and} \quad \Sigma_x = \begin{bmatrix}a^2 & 0\\

0 & b^2 \end{bmatrix}

$$



- Then, we can find $\Sigma_y$ such that

    - The equation of dependent ellipse ( Basis $\hat{y}$ )

    - $\Sigma_x$는 대각행렬이지만 (Independent하니까) $\Sigma_y$는 대각 행렬이 아니다. (Dependent하니까)



$$

\begin{align*}

y^\top  \Sigma_y^{-1}y &= 1 \quad \text{and} \quad \Sigma_y = ?\\\\

\implies x^\top  \Sigma_x^{-1} x &= y^\top  U\Sigma_x^{-1} U^\top y = 1 \quad

(\Sigma_y^{-1} \; \text{: similar matrix to } \Sigma_x^{-1}) \\\\

\therefore \;\;\Sigma_y^{-1} &= U\Sigma_x^{-1}U^\top  \; \text{or} \;\,

\Sigma_y = U\Sigma_x U^\top 

\end{align*}

$$



#### 2.4 Reverse Problem



- Given $\Sigma_y$

    - how to find $a$ (major axis) and $b$ (minor axis)

    - how to find the $\Sigma_x$

    - how to find the proper matrix $U$



$$

\begin{aligned}

\Sigma_x \quad &\underrightarrow{U} \quad \Sigma_y \\\\

\Sigma_y \quad &\underrightarrow{?} \quad \Sigma_x\end{aligned}

$$



- Eigenvectors of $\Sigma$



$$

A=S \Lambda S^\top \qquad \text{where  } S=[\upsilon_1 \; \upsilon_2]\, \text{ eigenvector of } A, \; \text{and }\Lambda=\begin{bmatrix} \lambda_1 & 0 \\ 0 & \lambda_2 \end{bmatrix} \\\,\\ \text{here, }\, \Sigma_y = U\Sigma_x U^\top = U \Lambda U^\top \qquad \text{where  } U = \begin{bmatrix}\hat{x}_1 & \hat{x}_2 \end{bmatrix}

\text{ eigenvector of } \Sigma_y, \; \text{and }\Lambda=\begin{bmatrix} \lambda_1 & 0 \\ 0 & \lambda_2 \end{bmatrix} = 

\begin{bmatrix} a^2 & 0 \\ 0 & b^2 \end{bmatrix}

$$



<center><img src = "image/5.png" width = "15%" height = "50%"></center>



$$

\text{Eigen-analysis}

\begin{cases}\Sigma_y\hat{x}_1=\lambda_1 \hat{x}_1\\

             \Sigma_y\hat{x}_2=\lambda_2 \hat{x}_2\\

\end{cases} \; \implies \;

 \Sigma_y \underbrace{\begin{bmatrix}\hat{x}_1 & \hat{x}_2 \end{bmatrix}}_{U} =

\underbrace{\begin{bmatrix} \hat{x}_1 & \hat{x}_2 \end{bmatrix}}_{U}

\underbrace{\begin{bmatrix}\lambda_1 & 0 \\ 0 & \lambda_2 \end{bmatrix}}_{\Lambda} ,\;\;\begin{matrix}\lambda_1 = a^2 \\ \lambda_2 = b^2 \end{matrix} \\ \\

\begin{aligned}

\Sigma_y U &= U\Lambda \\

\Sigma_y &= U \Lambda U^\top = U \Sigma_x U^\top

\end{aligned}

$$



$$

\begin{matrix} \, \\ x=U^\top y \\\\

\begin{bmatrix}x_1 \\ x_2 \end{bmatrix}

= U^\top \begin{bmatrix} y_1 \\ y_2 \end{bmatrix} \end{matrix}

\quad

\quad



\begin{matrix} \begin{aligned}

& a=\sqrt{\lambda_1}\\

& b = \sqrt{\lambda_2}\\

& \text{major axis}=\hat{x}_1, \\ &\text{minor axis}=\hat{x}_2

\end{aligned} \end{matrix}

$$



- Coordinate transformations to be independent by Eigen-analysis.

    - Since $\Sigma_y$ is given, we can calculate the eigenvalue, vector.

    - eigen vector : New basis.

    - eigen value : Major and minor axes of ellipses.


- - -


## 3. Gaussian Distribution

#### 3.1 Univariate Gaussian Distribution $\sim \mathcal{N}\left(\mu, \sigma^2\right)$

$$x \sim \mathcal{N}\left(\mu,\sigma^2\right) \;\;\to\;\; \mathcal{N}(x;\,\mu,\sigma) = \frac{1}{\sqrt{2\pi}\sigma}\exp\left(-\frac{1}{2}\frac{\left(x-\mu\right)^2}{\sigma^2}\right)$$



- Standardization

$$\mathcal{N}\left(\mu,\sigma^2\right) \;\; \to \;\; \mathcal{N}\left(0,1^2\right)$$

\\

$$

\begin{aligned}

&\triangleright\,1. \;\; \mathbb{E}\left[x\right] = \mu \\ 

&\;\;\;\;\;\;\;\;\, \mathbb{E}\left[x-\mu\right] = \mathbb{E}\left[x\right] - \mu = 0 \\ 

&\;\;\;\;\;\;\;\;\, \mathbb{E}\Big[\frac{x-\mu}{\sigma}\Big] = \frac{1}{\sigma}\,\mathbb{E}\left[x-\mu\right] = 0  \\\\



&\triangleright\,2. \;\; \text{Var}\left[x\right] = \sigma^2 \\ 

&\;\;\;\;\;\;\;\;\, \text{Var}\left[x - \mu \right] = \sigma^2 \\ 

&\;\;\;\;\;\;\;\;\, \text{Var}\Big[\frac{x - \mu}{\sigma} \Big] = \frac{\text{Var}\left[x-\mu\right]}{\sigma^2} = 1 \\\\



&\implies\;\; y = \frac{x-\mu}{\sigma}, \;x = \sigma y + \mu

\end{aligned}

$$


#### 3.2 Multivariate Gaussian Distribution



$$

\begin{align*} \mathcal{N}\big(x;\, \mu,\Sigma\big) &= \frac{1}{(2\pi)^{\frac{n}{2}}\lvert\Sigma\rvert^{\frac{1}{2}}}\exp\left(-\frac{1}{2}\left(x-\mu\right)^T\Sigma^{-1}\left(x-\mu\right)\right) \\ \\

\mu &= \text{length}\,\, n \,\,\text{column vector} \\

\Sigma &= n \times n \,\,\text{matrix (covariance matrix)} \\

\lvert\Sigma\rvert &= \text{matrix determinant}

\end{align*}

$$



- The contour of equal probability is ellipse



$$

\Delta^2 = (x-\mu)^T\Sigma^{-1} (x-\mu) = \text{const.} \;\; \to \;\; \text{ellipse}

$$


#### 3.3 Two Independent Variables

$$

\begin{align*}

P\left(x_1,x_2\right) &= P\left(x_1\right)P\left(x_2\right) \\

&= \frac{1}{(2\pi)^{\frac{n_1}{2}}\lvert\Sigma_{x_1}\rvert^{\frac{1}{2}}} \cdot \frac{1}{(2\pi)^{\frac{n_2}{2}}\lvert\Sigma_{x_2}\rvert^{\frac{1}{2}}} \cdot

\exp \Big(-\frac{1}{2} \frac{\left(x_1-\mu_{x_1}\right)^2}{\sigma_{x_1}^2}\Big) \cdot \exp \Big(-\frac{1}{2} \frac{\left(x_2-\mu_{x_2}\right)^2}{\sigma_{x_2}^2}\Big)

\\

&= \frac{1}{Z_1Z_2} \cdot \exp \Big(-\frac{1}{2}\Big(\frac{\left(x_1-\mu_{x_1}\right)^2}{\sigma_{x_1}^2} + \frac{\left(x_2-\mu_{x_2}\right)^2}{\sigma_{x_2}^2}\Big)\Big)\\

\end{align*}

$$



- In a matrix form

$$

P(x_1) \cdot P(x_2)=\frac{1}{Z_1Z_2}\exp\left(-\frac{1}{2}(x-\mu)^\top\Sigma^{-1}(x-\mu)\right)

$$



- Geometry of Gaussian

<center><img src = "image/6.png" width = "13%" height = "50%"></center>



$$

\begin{align*}\frac{x_1^2}{\sigma_{x_1}^2} + \frac{x_2^2}{\sigma_{x_2}^2} &= c \;\;\to\;\; \text{(ellipse)} \\ \\

\begin{bmatrix} x_1 & x_2\end{bmatrix} \begin{bmatrix} \frac{1}{\sigma_{x_1}^2} & 0 \\ 0 & \frac{1}{\sigma_{x_2}^2}

\end{bmatrix}

\begin{bmatrix}x_1 \\ x_2\end{bmatrix} &= c \qquad (\sigma_{x_1} < \sigma_{x_2})

\end{align*}

$$


#### 3.4 Two Dependent Variables

<center><img src = "image/7.png" width = "30%" height = "50%"></center>



- Compute  $P_Y(y)$ from $P_X(x)$

    - Basis를 변환해서 dependent하게 만들자



$$

\begin{aligned}

x &= \begin{bmatrix}\hat{x}_1 & \hat{x}_2 \end{bmatrix}^\top y = U^\top y \\\\

x^\top\Sigma_x^{-1} x &= y^\top U\Sigma_x^{-1}U^\top y = y^\top \Sigma_y^{-1} y \\

&\therefore \;\; \Sigma_y^{-1} = U\Sigma_x^{-1} U^\top\\

&\;\;\;\;\,\;\; \Sigma_y = U \Sigma_x U^\top

\end{aligned}

$$



- If $U$ is an eigenvector matrix of $\Sigma_y$, then $Σ_x$ is a diagonal matrix.



- In a Gaussian distribution, the information about whether the random variables are independent or dependent is contained in the covariance matrix.

    - Diagonal matrix $\;\;\to\;\;$ Independent

    - Otherewise $\;\;\to\;\;$ Dependent


- - -


## 4. Properties of Gaussian Distribution

- Symmetric about the mean



- Parameterized

    - The function estimation problem is simplified to a parameter estimation problem.

- Uncorrelated $\Longleftrightarrow$ Independent



- Gaussian distributions are closed to

  - Linear transformation

  

  - Affine transformation

  - Reduced dimension of multivariate Gaussian

    - Marginalization (projection)

    

    - Conditioning (slice)



<center><img src = "image/8.png" width = "50%" height = "50%"></center>


#### 4.1 Affine Transformation

$$

\begin{aligned}

x &\sim \mathcal{N}(\mu_x, \Sigma_x) \\

y &= Ax + b\\

\end{aligned}

$$



$$

\begin{aligned}

\mathbb{E}[y] & = \mathbb{E}[Ax + b] = A\mathbb{E}[x] + b = A\mu_x + b\\

\text{cov}(y) & = \text{cov}(Ax + b) = \text{cov}(Ax) = A\text{cov}(x)A^\top = A\Sigma_x A^\top \\\\

&\therefore y \sim \mathcal{N}\left(A\mu_x+b, A\Sigma_x A^\top  \right)

\end{aligned}

$$


#### 4.2 Marginal Probability of a Gaussian



$$

x \sim \mathcal{N}(\mu, \Sigma), \;\;\;\; x = \begin{bmatrix} x_1 \\ x_2\end{bmatrix}, \;\; \mu = \begin{bmatrix} \mu_1 \\ \mu_2\end{bmatrix}, \;\; \Sigma = \begin{bmatrix} \Sigma_{11}& \Sigma_{12}\\ \Sigma_{21}& \Sigma_{22} \end{bmatrix}

$$



$$

\begin{aligned} \\ 

x_1 &= \begin{bmatrix} I & 0\end{bmatrix} x = Ax \;\; \to \;\; \text{affine transformation} \\

\mathbb{E}[x_1] &= \begin{bmatrix} I & 0\end{bmatrix} \mathbb{E}[x] = \mu_1 \\

\text{cov}(x_1) &= \begin{bmatrix} I & 0\end{bmatrix} \text{cov}(x) \begin{bmatrix} I \\ 0\end{bmatrix} = \begin{bmatrix} I & 0\end{bmatrix}\Sigma \begin{bmatrix} I \\ 0\end{bmatrix} = \Sigma_{11}

\\\\

&\therefore x_1 \sim \mathcal{N}\left(\mu_1, \Sigma_{11} \right)

\end{aligned}

$$


#### 4.3. Component of a Gaussian Random Vector



- Suppose $x \sim \mathcal{N}(0, \Sigma),\; c \in \mathbb{R}^n$ be a unit vector

$$y = c^\top x$$



- $y$ is the component of $x$ in the direction $c$ (Inner product)



- $y$ is Gaussian

$$\begin{aligned} \mathbb{E}[y]&=0 \\ \text{cov}(y) &= \mathbb{E}[y^2] - \mathbb{E}[y]^2 = \mathbb{E}[y^2] - 0 \\ &= \mathbb{E}[c^\top x x^\top c] \\ &= c^\top Σc \end{aligned}$$

 

- The unit vector $c$ that minimizes $c^\topΣc$ is the eigenvector of $Σ$ with the smallest eigenvalue



  - PCA

  

  - $\mathbb{E}[y^2]=λ_{min}$

 


#### 4.4. Conditional Probability of a Gaussian Random Vector



$$

\begin{bmatrix} x \\ y\end{bmatrix} \sim \mathcal{N}\left( \begin{bmatrix} \mu_x \\ \mu_y\end{bmatrix}, \begin{bmatrix} \Sigma_{x}& \Sigma_{xy}\\ \Sigma_{yx}& \Sigma_{y} \end{bmatrix} \right)

$$



- The conditional pdf of $x$ given $y$ is Gaussian

$$

x \mid y \sim \mathcal{N} \left(\mu_x + \Sigma_{xy}\Sigma_{y}^{-1}(y-\mu_y), \; \Sigma_{x} - \Sigma_{xy}\Sigma_{y}^{-1}\Sigma_{yx} \right)

$$



- Notice that conditional confidence intervals are narrower. i.e., measuring  $y$ gives information about $x$

$$

\text{cov}(x \mid y) = \Sigma_{x} - \Sigma_{xy}\Sigma_{y}^{-1}\Sigma_{yx} \leq \Sigma_{x}

$$

