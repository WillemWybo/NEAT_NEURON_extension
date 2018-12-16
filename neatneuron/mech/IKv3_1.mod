: This mod file is automaticaly generated by the ``neat.channels.ionchannels`` module

NEURON {
    SUFFIX IKv3_1
    USEION k WRITE ik
    RANGE  g, e
    GLOBAL var0inf, tau0
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
    ik (mA/cm2)
    var0inf
    tau0 (ms)
    v (mV)
}

STATE {
    var0
}

BREAKPOINT {
    SOLVE states METHOD cnexp
    ik = g * ( var0 *1.0) * (v - e)
}

INITIAL {
    rates(v)
    var0 = var0inf
}

DERIVATIVE states {
    rates(v)
    var0' = (var0inf - var0) / tau0
}

PROCEDURE rates(v) {
    var0inf = 1.0/(exp(-0.10309278350515465*v + 1.9278350515463918) + 1.0)
    tau0 = 4.0/(exp(-0.022655188038060714*v - 1.0548255550521068) + 1.0)
}

