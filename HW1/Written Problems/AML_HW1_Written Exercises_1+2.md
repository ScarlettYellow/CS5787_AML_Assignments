

1.
$$
\begin{equation}
	\mathop{\arg\max}\limits_{\theta} E_{\hat{p}(x,y))}[logP_{\theta}(y|x))] 
\end{equation}
$$

$$
\begin{equation}
	=\mathop{\arg\min}\limits_{\theta} E_{\hat{p}(x)}[KL(\hat{p}(y|x)||P_{\theta}(y|x)] 
\end{equation}
$$

$$
\begin{equation}
	=\mathop{\arg\min}\limits_{\theta} E_{\hat{p}(x)}[E_{\hat{p}(x,y)}[log{\hat{p}(y|x)} - logP_{\theta}(y|x)]] 
\end{equation}
$$

As is the empirical data distribution (no randomness), $E_{\hat{p}(x,y)} log{\hat{p}(y|x)}$ is Constant
$$
\begin{equation}
	= C - \mathop{\arg\min}\limits_{\theta}  E_{\hat{p}(x)}E_{\hat{p}(x,y)}(- logP_{\theta}(y|x))
\end{equation}
$$

$$
\begin{equation}
	= C - \mathop{\arg\min}\limits_{\theta} E_{\hat{p}(x,y)}(- logP_{\theta}(y|x))
\end{equation}
$$

$$
\begin{equation}
	= C + \mathop{\arg\max}\limits_{\theta} E_{\hat{p}(x,y)} logP_{\theta}(y|x)
\end{equation}
$$



2.

(a) suppose: 

- test quality as event A
- actual quality as event B
- defective: 1
- not defective: 0

- test defective: $P(A=1)$
- test not defective: $P(A=0)$
- actually defective: $P(B=1)$
- actually not defective: $P(B=0)$

from the text we know:

- $P(A=1|B=1) = P(A=0|B=0) = 0.95$
- $P(A=0|B=1) = P(A=1|B=0) = 0.05$
- $P(B=1)=\frac{1}{100000}=0.00001$
- $P(B=0)=1-P(B=1)=0.99999$

we can calculate out:

- $P(A=1) = P(A=1|B=1) P(B=1) + P(A=1|B=0) P(B=0)= 0.95*0.00001+0.05*0.99999 = 0.050009$
- $P(A=0) = 1-0.050009 = 0.949991$

the chances that the widge is actually defective given the test defective result:
$$
P(B=1|A=1) = \frac{P(B=1)P(A=1|B=1)}{P(A=1)} = \frac{0.00001*0.95}{0.050009}=0.000189966
$$


(b) sum widgets per year = 10000000

the probability of good widgets are thrown away per year:
$$
P(B=0|A=1) = 1-P(B=1|A=1) = 1 - 0.000189966 = 0.999810034
$$
the number of good widgets are thrown away per year:
$$
10000000 * P(B=0|A=1) * P(A=1)
$$

$$
= 10000000 * 0.999810034 *0.050009 = 499995
$$

the probability of bad widgets are still shipped to customers each year:
$$
P(B=1|A=0) = \frac{P(B=1)P(A=0|B=1)}{P(A=0)}
$$
the number of bad widgets are still shipped to customers each year:
$$
10000000 * P(B=1|A=0) * P(A=0)
$$

$$
= 10000000 *  \frac{P(B=1)P(A=0|B=1)}{P(A=0)} * P(A=0)
$$

$$
= 10000000 * \frac{0.00001 * 0.05}{0.949991} * 0.949991 = 5
$$

