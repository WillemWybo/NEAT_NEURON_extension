: This mod file is automaticaly generated by the ``neat.channels.ionchannels`` module

NEURON {
    SUFFIX Ih
    NONSPECIFIC_CURRENT i
    RANGE  g, e
    GLOBAL var0inf, var1inf, tau0, tau1
    THREADSAFE
}

PARAMETER {
    g = 0.0 (S/cm2)
    e = 0.0 (mV)
}

UNITS {
    (mA) = (milliamp)
    (mV) = (millivolt)
    (mM) = (milli/liter)
}

ASSIGNED {
    i (mA/cm2)
    var0inf
    tau0 (ms)
    var1inf
    tau1 (ms)
    v (mV)
}

STATE {
    var0
    var1
}

BREAKPOINT {
    SOLVE states METHOD cnexp
    i = g * ( var0 *0.8 +  var1 *0.2) * (v - e)
}

INITIAL {
    rates(v)
    var0 = var0inf
    var1 = var1inf
}

DERIVATIVE states {
    rates(v)
    var0' = (var0inf - var0) / tau0
    var1' = (var1inf - var1) / tau1
}

PROCEDURE rates(v) {
    var0inf = 1.0/(exp(0.14285714285714285*v + 11.714285714285714) + 1.0)
    tau0 = 40.0
    var1inf = 1.0/(exp(0.14285714285714285*v + 11.714285714285714) + 1.0)
    tau1 = 300.0
}

