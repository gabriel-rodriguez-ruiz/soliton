#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 12:06:00 2023

@author: gabriel
"""

import sympy as sp

x = sp.Symbol("x")
integral = sp.integrate(sp.sin(sp.tanh(x)), (x, 0, 1))