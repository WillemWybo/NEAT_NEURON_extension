: This mod file is automaticaly generated by the ``neat.channels.ionchannels`` module

NEURON {
    SUFFIX INa_p
    USEION na WRITE ina
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
    ina (mA/cm2)
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
    ina = g * ( var0 * var0 * var0 * var1 *1.0) * (v - e)
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
    var0inf = 1.0/(exp(-0.21739130434782611*v - 11.434782608695654) + 1.0)
    tau0 = 2.0338983050847457/((-0.124*v - 4.7119999999999997)/(-563.03023683595109*exp(0.16666666666666666*v) + 1.0) + (0.182*v + 6.9159999999999995)/(1.0 - 0.0017761035457343791*exp(-0.16666666666666666*v)))
    var1inf = 1.0/(exp(0.10000000000000001*v + 4.8799999999999999) + 1.0)
    tau1 = 0.33898305084745761/((-2.88e-6*v - 4.8959999999999999e-5)/(-39.318937124774365*exp(0.21598272138228941*v) + 1.0) + (6.9399999999999996e-6*v + 0.00044693599999999999)/(1.0 - 2.320410263420138e-11*exp(-0.38022813688212931*v)))
}

