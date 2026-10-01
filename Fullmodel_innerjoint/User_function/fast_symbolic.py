# -*- coding: utf-8 -*-
"""Compile Robotran's generated symbolic functions with numba.

The generated files in symbolicR/ (direct dynamics, kinematics of the contact points
and sensors) are long sequences of float operations; run by CPython they take most of
the simulation time. accelerate(mbs_data), called from user_dirdyn_init, translates
them into numba kernels and installs those on mbs_data, which MBsysPy calls instead
of the Python functions. The generated files are not modified: the translated
sources are written to symbolicR/__numba__/ (named after a hash of the original, so
they follow regenerated symbolic files) and numba caches their machine code there.

Without numba, or if a generated file does not have the expected layout, the
original Python functions are kept.
"""
import hashlib
import importlib.util
import os
import re
import sys

try:
    import numba  # noqa: F401
except ImportError:
    numba = None

S_ARRAYS = ("dpt", "l", "m", "In", "g", "frc", "trq")


def accelerate(mbs_data, verbose=True):
    """Install numba versions of dirdyna, extforces and sensor on mbs_data (if possible)."""
    if numba is None:
        if verbose:
            print("fast_symbolic: numba is not installed, using the Python symbolic files")
        return False
    done = []
    for name, translate, attribute in (("dirdyna", _translate_dirdyna, "mbs_dirdyna"),
                                       ("extforces", _translate_extforces, "mbs_extforces"),
                                       ("sensor", _translate_sensor, "mbs_sensor")):
        path = os.path.join(mbs_data.symbolic_path, "mbs_%s_%s.py" % (name, mbs_data.mbs_name))
        try:
            module = _load(path, name, translate)
            setattr(mbs_data, attribute, getattr(module, name))
            done.append(name)
        except Exception as error:  # keep the Python version of this function
            if verbose:
                print("fast_symbolic: kept the Python %s (%s)" % (name, error))
    return len(done) == 3


def _load(path, name, translate):
    with open(path) as f:
        source = f.read()
    digest = hashlib.sha1((source + _VERSION).encode()).hexdigest()[:12]
    cache_dir = os.path.join(os.path.dirname(os.path.abspath(path)), "__numba__")
    target = os.path.join(cache_dir, "fast_%s_%s.py" % (name, digest))
    if not os.path.exists(target):
        os.makedirs(cache_dir, exist_ok=True)
        tmp = "%s.%d.tmp" % (target, os.getpid())
        with open(tmp, "w") as f:
            f.write(translate(source))
        os.replace(tmp, target)  # atomic: safe with parallel simulations
    module_name = "fast_%s_%s" % (name, digest)
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, target)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module  # numba's cache resolves the kernels by module name
    spec.loader.exec_module(module)
    return module


_VERSION = "2"  # bump when the translation below changes

HEADER = "from numba import njit\nfrom numpy import zeros\nfrom math import sin, cos, sqrt\n\n"


def _body(source, signature):
    """Lines of the generated function after its def line."""
    lines = source.splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith(signature))
    return lines[start + 1:]


def _strip_aliases(lines, aliases):
    """Drop 'x = s.x' alias lines (the arrays become kernel arguments)."""
    pattern = re.compile(r"^\s*(%s)\s*=\s*s\.\1\s*$" % "|".join(aliases))
    return [line for line in lines if not pattern.match(line)]


def _no_leftover(code, what):
    left = re.findall(r"\b(?:s|sens)\.\w+", code)
    if left:
        raise ValueError("unexpected references in %s: %s" % (what, sorted(set(left))[:5]))


def _translate_dirdyna(source):
    body = _strip_aliases(_body(source, "def dirdyna(M, c, s, tsim):"), ("q", "qd"))
    code = "\n".join(body)
    code = re.sub(r"\bs\.(%s)\b" % "|".join(S_ARRAYS), r"\1", code)
    _no_leftover(code, "dirdyna")
    return (HEADER + "@njit(cache=True)\ndef kernel(M, c, q, qd, dpt, l, m, In, g, frc, trq):\n" + code + "\n\n"
            "def dirdyna(M, c, s, tsim):\n"
            "    kernel(M, c, s.q, s.qd, s.dpt, s.l, s.m, s.In, s.g, s.frc, s.trq)\n")


def _translate_extforces(source):
    body = _strip_aliases(_body(source, "def extforces(frc, trq, s, tsim):"), ("q", "qd", "qdd", "frc", "trq"))
    calls = [i for i, line in enumerate(body) if "s.user_ExtForces(" in line]
    if calls != list(range(calls[0], calls[-1] + 1)):
        raise ValueError("user_ExtForces calls are not consecutive")
    kinematics, user_calls, mapping = body[:calls[0]], body[calls[0]:calls[-1] + 1], body[calls[-1] + 1:]
    names = lambda lines: set(re.findall(r"\b([A-Za-z_]\w*)\b", "\n".join(lines)))
    assigned = lambda code: set(re.findall(r"^\s*([A-Za-z_]\w*)\s*(?:\[[^\]]*\])?\s*=", code, re.M))

    # 1. contact-point kinematics -> kernel returning what the user calls and the mapping need
    kin_code = re.sub(r"\bs\.(%s)\b" % "|".join(S_ARRAYS), r"\1", "\n".join(kinematics))
    _no_leftover(kin_code, "extforces kinematics")
    shared = sorted(assigned(kin_code) & (names(user_calls) | names(mapping)))
    if not shared:
        raise ValueError("no contact kinematics found")

    # 2. contact forces (SWr) -> generalized forces on the bodies, also a kernel
    map_code = re.sub(r"\bs\.(%s)\b" % "|".join(S_ARRAYS), r"\1", "\n".join(mapping))
    _no_leftover(map_code, "extforces mapping")
    swr = sorted(assigned("\n".join(user_calls)))
    map_inputs = sorted((set(shared) & names(mapping)) | set(swr))

    return (HEADER + "@njit(cache=True)\ndef kinematics(q, qd, qdd, dpt, l, m, In, g, frc, trq):\n" + kin_code +
            "\n    return (%s,)\n\n" % ", ".join(shared) +
            "@njit(cache=True)\ndef mapping(%s, l, frc, trq):\n" % ", ".join(map_inputs) + map_code + "\n\n" +
            "def extforces(frc, trq, s, tsim):\n"
            "    (%s,) = kinematics(s.q, s.qd, s.qdd, s.dpt, s.l, s.m, s.In, s.g, s.frc, s.trq)\n" % ", ".join(shared) +
            "\n".join(user_calls) + "\n"
            "    mapping(%s, s.l, s.frc, s.trq)\n" % ", ".join(map_inputs))


def _translate_sensor(source):
    body = _strip_aliases(_body(source, "def sensor(sens, s, isens):"), ("q", "qd", "qdd", "dpt"))
    code = "\n".join(body)
    code = re.sub(r"\bsens\.(OMP|OM|P|R|V|A|J)\b", r"s_\1", code)
    code = re.sub(r"\bs\.(%s)\b" % "|".join(S_ARRAYS), r"\1", code)
    _no_leftover(code, "sensor")
    return (HEADER + "@njit(cache=True)\n"
            "def kernel(s_P, s_R, s_V, s_OM, s_A, s_OMP, s_J, q, qd, qdd, dpt, isens):\n" + code + "\n\n"
            "def sensor(sens, s, isens):\n"
            "    kernel(sens.P, sens.R, sens.V, sens.OM, sens.A, sens.OMP, sens.J, s.q, s.qd, s.qdd, s.dpt, isens)\n")
