目录

[TOC]

# Markdown MathTyping

Two ways of add a new equation in markdown:

- Inline equation with `$ ... $`: $\Gamma(z) = xxx $
- Independent equation with `$$ ... $$`: 

$$
\Gamma(z) = xxx
$$

## Greece alphabet

|  名称   |   大写    |  code   |    小写    |   code   |
| :-----: | :-------: | :-----: | :--------: | :------: |
|  Alpha  |    $A$    |    A    |  $\alpha$  |  \alpha  |
|  Beta   |    $B$    |    B    |  $\beta$   |  \beta   |
|  Gamma  | $\Gamma$  | \Gamma  |  $\gamma$  |  \gamma  |
|  Delta  | $\Delta$  | \Delta  |  $\delta$  |  \delta  |
| Epsilon |    $E$    |    E    | $\epsilon$ | \epsilon |
|  Zeta   |    $Z$    |    Z    |  $\zeta$   |  \zeta   |
|   Eta   |    $H$    |    H    |   $\eta$   |   \eta   |
|  Theta  | $\Theta$  | \Theta  |  $\theta$  |  \theta  |
|  Iota   |    $I$    |    I    |  $\iota$   |  \iota   |
|  Kappa  |    $K$    |    K    |  $\kappa$  |  \kappa  |
| Lambda  | $\Lambda$ | \Lambda | $\lambda$  | \lambda  |
|   Mu    |    $M$    |    M    |   $\mu$    |   \mu    |
|   Nu    |    $N$    |    N    |   $\nu$    |   \nu    |
|   Xi    |   $\Xi$   |   \Xi   |   $\xi$    |   \xi    |
| Omicron |    $O$    |    O    | $\omicron$ | \omicron |
|   Pi    |   $\Pi$   |   \Pi   |   $\pi$    |   \pi    |
|   Rho   |    $P$    |    P    |   $\rho$   |   \rho   |
|  Sigma  | $\Sigma$  | \Sigma  |  $\sigma$  |  \sigma  |
|   Tau   |    $T$    |    T    |   $\tau$   |   \tau   |
| Upsilon |    $Y$    |    Y    |            |          |
|   Phi   |  $\Phi$   |  \Phi   |   $\phi$   |   \phi   |
|   Chi   |    $X$    |    X    |   $\chi$   |   \chi   |
|   Psi   |  $\Psi$   |  \Psi   |   $\psi$   |   \psi   |
|  Omega  | $\Omega$  | \Omega  |  $\omega$  |  \omega  |

## Parentheses

- `$\left(\frac{x}{y}\right)$`: $\left(\frac{x}{y}\right)$

- `$\lbrace a+b\rbrace$`: $\lbrace a+b\rbrace$
- `$\langle x \rangle$`: $\langle x \rangle$
- `$\lceil x \rceil$`: $\lceil x \rceil$
- `$\lfloor x \rfloor$`: $\lfloor x \rfloor$

## Sum and Integral

- `$\sum_{r=1}^n$`: $\sum_{r=1}^n$
- `$\int_{r=1}^\infty$`: $\int_{r=1}^\infty$
- `$\iint$`: $\iint$
- `$\iiint$`: $\iiint$
- `$\iiiint$`: $\iiiint$
- `$\prod {a+b}$`: $\prod {a+b}$
- `$\prod_{i=1}^{K}$`: $\prod_{i=1}^{K}$
- `$\prod$`: $\prod$
- `$\bigcup$`: $\bigcup$
- `$\bigcap$`: $\bigcap$
- `$arg\,\max_{c_k}$`: $arg\,\max_{c_k}$
- `$arg\,\min_{c_k}$`: $arg\,\min_{c_k}$
- `$\mathop {argmin}_{c_k}$`: $\mathop {argmin}_{c_k}$
- `$\mathop {argmax}_{c_k}$`: $\mathop {argmax}_{c_k}$
- `$\max_{c_k}$`: $\max_{c_k}$
- `$\min_{c_k}$`: $\min_{c_k}$

## Sqrt and divide

- `$\frac {a+c+1}{b+c+2}$`: $\frac {a+c+1}{b+c+2}$
- `${a+1\over b+1}`: ${a+1\over b+1}$

- `$$x=a_0 + \frac {1^2}{a_1 + \frac {2^2}{a_2 + \frac {3^2}{a_3 + \frac {4^2}{a_4 + ...}}}}$$`: 

$$
$$x=a_0 + \frac {1^2}{a_1 + \frac {2^2}{a_2 + \frac {3^2}{a_3 + \frac {4^2}{a_4 + ...}}}}$$
$$

- `$\sqrt[4]{\frac xy}$`: $\sqrt[4]{\frac xy}$
- `$\sqrt {a+b}$`: $\sqrt {a+b}$

## Multiline

- `\begin{cases} ... \end{cases}`:
  - `\\` : switch line
  - `&`: alignment
  - `\   `: space/brank

$$
L(Y,f(X)) =
\begin{cases}
0, & \text{Y = f(X)} \\[5ex]
1, & \text{Y $\neq$ f(X)}
\end{cases}
$$

- `\begin{equation}\begin{split} ... \end{split}\end{equation}`

$$
\begin{equation}\begin{split} 
a&=b+c-d \\ 
&\quad +e-f\\ 
&=g+h\\ 
& =i 
\end{split}\end{equation}
$$

- `\left \{ \begin{array}{c} ...\end{array}\right.`

$$
\left \{ 
\begin{array}{c}
a_1x+b_1y+c_1z=d_1 \\ 
a_2x+b_2y+c_2z=d_2 \\ 
a_3x+b_3y+c_3z=d_3
\end{array}
\right.
$$

## Special character

### Trigonometric

- `\sinx$`: $\sin x$
- `\arctan x`: $\arctan x$

### Compare

- `\lt`: $\lt$
- `\gt`: $\gt$
- `\le`: $\le$
- `\ge`: $\ge$
- `\ne`: $\ne$
- `\not\lt`: $\not\lt$

### Set

- `\cup`: $\cup$
- `\cap`: $\cap$
- `\setminus`: $\setminus$
- `\subset`: $\subset$
- `\subseteq`: $\subseteq$
- `\subsetneq`: $\subsetneq$
- `\supset`: $\supset$
- `\in`: $\in$
- `\notin`: $\notin$
- `\emptyset`: $\emptyset$
- `\varnothing`: $\varnothing$

### Array

- `\binom{n+1}{2k}`: $\binom{n+1}{2k}$
- `{n+1 \choose 2k}`: ${n+1 \choose 2k}$

### Arrow

- `\to`: $\to$
- `\rightarrow`: $\rightarrow$
- `\leftarrow`: $\leftarrow$
- `\Rightarrow`: $\Rightarrow$
- `\Leftarrow`: $\Leftarrow$
- `\mapsto`: $\mapsto$

### Logistic

- `\land`: $\land$
- `\lor`: $\lor$
- `\lnot`: $\lnot$
- `\forall`: $\forall$
- `\exists`: $\exists$
- `\top`: $\top$
- `\bot`: $\bot$
- `\vdash`: $\vdash$
- `\vDash`: $\vDash$

### Operation

- `\star`: $\star$
- `\ast`: $\ast$
- `\oplus`: $\oplus$
- `\approx`: $\approx$
- `\sim`: $\sim$
- `\equiv`: $\equiv$
- `\prec`: $\prec$

### Range

- `\infty`: $\infty$
- `\aleph_o`: $\aleph_o$
- `\nabla`: $\nabla$
- `\Im`: $\Im$
- `\Re`: $\Re$

### Range and Dot

- `b\pmod n`: $b\pmod n$
- `a \equiv b \pmod n`: $a \equiv b \pmod n$
- `\ldots`: $\ldots$
- `\cdots`: $\cdots$
- `\cdot`: $\cdot$

## Table

- `\begin{array}{列样式}…\end{array}`
  - `clr`: central
  - `|`: horizantal line
  - `\\`: seperation of rows
  - `&`: seperation of colunms
  - `\hline`: vertical line

$$
\begin{array}{c|lcr}
n & \text{Left} & \text{Center} & \text{Right} \\
\hline
1 & 0.24 & 1 & 125 \\
2 & -1 & 189 & -8 \\
3 & -20 & 2000 & 1+10i \\
\end{array}
$$

## Upper character

- `\hat x`: $\hat x$
- `\widehat {xy}`: $\widehat {xy}$
- `\overline x`: $\overline x$
- `\vec`: $\vec x$
- `\overrightarrow {xy}`: $\overrightarrow {xy}$
- `\dot x`: $\dot x$
- `\ddot x`: $\ddot x$
- `\dot {\dot x}`: $\dot {\dot x}$

## Matrix

- `\begin{matrix}…\end{matrix}`

$$
\begin{matrix}
1 & x & x^2 \\
1 & y & y^2 \\
1 & z & z^2 \\
\end{matrix}
$$

- `$\begin{pmatrix}1 & 2 \\ 3 & 4\\ \end{pmatrix}$`:$\begin{pmatrix}1 & 2 \\ 3 & 4\\ \end{pmatrix}$

- `$\begin{bmatrix}1 & 2 \\ 3 & 4\\ \end{bmatrix}$`: $\begin{bmatrix}1 & 2 \\ 3 & 4\\ \end{bmatrix}$
- `$\begin{Bmatrix}1 & 2 \\ 3 & 4\\ \end{Bmatrix}$`: $\begin{Bmatrix}1 & 2 \\ 3 & 4\\ \end{Bmatrix}$
- `$\begin{vmatrix}1 & 2 \\ 3 & 4\\ \end{vmatrix}$`: $\begin{vmatrix}1 & 2 \\ 3 & 4\\ \end{vmatrix}$
- `$\begin{Vmatrix}1 & 2 \\ 3 & 4\\ \end{Vmatrix}$`: $\begin{Vmatrix}1 & 2 \\ 3 & 4\\ \end{Vmatrix}$
- ignore elements using `\cdots \vdots \ddots` : 

$$
\begin{pmatrix}
1&a_1&a_1^2&\cdots&a_1^n\\
1&a_2&a_2^2&\cdots&a_2^n\\
\vdots&\vdots&\vdots&\ddots&\vdots\\
1&a_m&a_m^2&\cdots&a_m^n\\
\end{pmatrix}
$$

- `\begin{array} ... \end{array}`: 

$$
\left[  \begin{array}  {c c | c} %这里的c表示数组中元素对其方式：c居中、r右对齐、l左对齐，竖线表示2、3列间插入竖线
1 & 2 & 3 \\
\hline %插入横线，如果去掉\hline就是增广矩阵
4 & 5 & 6
\end{array}  \right]
$$



## Equation Quote

### Tag

- Using `\tag{yourtag}` to tag the equation
- If you want to quote the equation in other equation, also mark the equation with corresponding label: `\label{yourlabel}` , for eg. `a = x^2 - y^3 \tag{1}\label{1}`

$$
a = x^2 - y^3 \tag{1}\label{1}
$$

### Quote

- `$a + y^3 \stackrel{\eqref{1}}= x^2$`: $a + y^3 \stackrel{\eqref{1}}= x^2$
- `$a + y^3 \stackrel{\ref{111}}= x^2$`: $a + y^3 \stackrel{\ref{111}}= x^2$

## Type and font

- `$\mathbb {ABCDE}F$`: $\mathbb {ABCDEF}$

- `$\Bbb {ABCDEF}$`: $\Bbb {ABCDEF}$
- `$\mathbf {ABCDEFGHIJKLMNOPQRSTUVWXYZ}$`: $\mathbf {ABCDEFGHIJKLMNOPQRSTUVWXYZ}$
- `$\mathbf {abcdefghijklmnopqrstuvwxyz}$`: $\mathbf {abcdefghijklmnopqrstuvwxyz}$
- `$\mathtt {ABCDEFGHIJKLMNOPQRSTUVWXYZ}$`: $\mathtt {ABCDEFGHIJKLMNOPQRSTUVWXYZ}$

## References

- [markdown中公式编辑教程](https://www.jianshu.com/p/25f0139637b7)
- [Mathjax公式教程](https://blog.csdn.net/dabokele/article/details/79577072)
- [基本数学公式语法](https://blog.csdn.net/ethmery/article/details/50670297)
- [常用数学符号的LaTeX表示方法](https://blog.csdn.net/lilongsy/article/details/79378620)
- [Beautiful math in all browsers](https://www.mathjax.org/)

