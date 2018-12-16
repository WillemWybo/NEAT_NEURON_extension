: This mod file is automaticaly generated by the ``neat.channels.ionchannels`` module

NEURON {
    SUFFIX Im
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
    var0inf = 0.10928099146368463*exp(0.10000000000000001*v)/(0.0033*33.115451958692312*exp(0.10000000000000001*v) + 0.0033*0.030197383422318501*exp(-0.10000000000000001*v))
    tau0 = 0.33898305084745761/(0.0033*33.115451958692312*exp(0.10000000000000001*v) + 0.0033*0.030197383422318501*exp(-0.10000000000000001*v))
}

