---
layout: post
title:  "Using deep neural networks to compute the mass of forming planets"
categories: [Co-Author]
tags: [Y. Alibert, J. Venturini]
---

Context: Computing the mass of planetary envelopes and the critical mass beyond which planets accrete gas in a runaway fashion is important for studying planet formation, in particular, for planets up to the Neptune-mass range. This computation in principle requires solving a set of differential equations, the internal structure equations, for some boundary conditions (pressure, temperature in the protoplanetary disc where a planet forms, core mass, and the rate of accretion of solids by the planet). Solving these equations in turn proves to be time-consuming and sometimes numerically unstable.

Aims: The aim is to provide a way to approximate the result of integrating the internal structure equations for a variety of boundary conditions.

Methods: We computed a set of internal planetary structures for a very large number (millions) of boundary conditions, considering two opacities: that of the interstellar medium, and a reduced opacity. This database was then used to train deep neural networks (DNN) in order to predict the critical core mass and the mass of planetary envelopes as a function of the boundary conditions.

Results: We show that our neural networks provide a very good approximation (at the percent level) of the result obtained by solving interior structure equations, but the required computer time is much shorter. The difference with the real solution is much smaller than the difference that is obtained with some analytical formulas that are available in the literature, which only provide the correct order of magnitude at best. We compare the results of the DNN with other popular machine-learning methods (random forest, gradient boost, support vector regression) and show that the DNN outperforms these methods by a factor of at least two.

Conclusions: We show that some analytical formulas that can be found in various papers can severely overestimate the mass of planets and therefore predict the formation of planets in the Jupiter-mass regime instead of the Neptune-mass regime. The python tools that we provide allow computing the critical mass and the mass of planetary envelopes in a variety of cases, without the requirement of solving the internal structure equations. These tools can easily replace previous analytical formulas and provide far more accurate results. 

---

Publisher-link: [https://doi.org/10.1051/0004-6361/201834942](https://doi.org/10.1051/0004-6361/201834942)

Arxiv-link: [https://arxiv.org/abs/1903.00320](https://arxiv.org/abs/1903.00320)

---