"""
Pulse sequence simulation code has been taken from VeSPA and implemented here with minor modifications.
"""

import pickle
import pygamma as pg
import numpy as np
from Utilities.constants import OMEGA


def fid(spin_system):
    sys_hamiltonian = pg.Hcs(spin_system) + pg.HJ(spin_system)
    read = pg.Fm(spin_system, "1H")
    ACQ = pg.acquire1D(pg.gen_op(read), sys_hamiltonian, 0.000001)

    sigma = pg.sigma_eq(spin_system)
    sigma0 = pg.Ixpuls(spin_system, sigma, "1H", 90.0)

    return pg.TTable1D(ACQ.table(sigma0))


def steam(spin_system, te=20, tm=10):
    # ------------------------------------------------------------------------
    # This is an example PyGAMMA pulse sequence for use in Vespa-Simulation
    #
    # A timing diagram for this pulse sequence can be found in the Appendix
    # of the Simulation User Manual.
    # ------------------------------------------------------------------------

    # the isotope string used to sort/select the metabolites of interest is passed
    # in the sim_desc object so the user can tailor other object within their
    # code to be nuclei specific, such as the observe operator or pulses
    obs_iso = '1H'

    # extract the dynamically changing variable from loops 1 and 2 for 'te'
    # and 'tm', divide by 1000.0 because the GUI states that values are
    # entered in [ms], but PyGAMMA wants [sec]

    te = te / 1000.0
    tm = tm / 1000.0

    # set up steady state and observation variables
    H = pg.Hcs(spin_system) + pg.HJ(spin_system)
    D = pg.Fm(spin_system, obs_iso)
    ac = pg.acquire1D(pg.gen_op(D), H, 0.000001)
    ACQ = ac

    # excite, propagate, refocus and acquire the data
    #
    # for the case of STEAM, we need to simulate crusher gradients around the
    # second and third 90 pulses. We do this by creating 4 copies of the
    # operator matrix at that point, rotate respectively  by 0, 90, 180 and 270
    # degrees to each other, apply the 90 pulses and TM period, and then
    # add the four back into one normalized matrix.

    dephase_ang = [0.0, 90.0, 180.0, 270.0]
    Udelay1 = pg.prop(H, te * 0.5)
    Udelay2 = pg.prop(H, tm)

    sigma0 = pg.sigma_eq(spin_system)
    # first 90 pulse, excite spins
    sigma0 = pg.Iypuls(spin_system, sigma0, obs_iso, 90.0)
    # nutate TE/2
    sigma0 = pg.evolve(sigma0, Udelay1)

    # Now we need to create the effect of crushers around the 2nd and 3rd
    # 90 pulses. This is done by creating 4 copies of spin state and repeating
    # the rest of the sequence for four different rotations around z-axis
    sigma_mult = []
    for i in dephase_ang:
        sigma_mult.append(pg.gen_op(sigma0))

    for i, angle in enumerate(dephase_ang):
        # calculate and apply rotation around z-axis
        riz = pg.gen_op(pg.Rz(spin_system, angle))
        sigma_mult[i] = pg.evolve(sigma_mult[i], riz)

        # second 90 pulse
        sigma_mult[i] = pg.Ixpuls(spin_system, sigma_mult[i], obs_iso, 90.0)
        # this function removes all coherences still in transverse plane
        # this removes all stimulated echos from first and second 90 pulse
        pg.zero_mqc(spin_system, sigma_mult[i], 0, -1)
        # third 90 pulse
        sigma_mult[i] = pg.Ixpuls(spin_system, sigma_mult[i], obs_iso, 90.0)
        # undo rotation around z-axis
        sigma_mult[i] = pg.evolve(sigma_mult[i], riz)
        # scale results based on the number of phase angles
        sigma_mult[i] *= 1.0 / float(len(dephase_ang))

        # sum up each rotated/unrotated results
        if i == 0:
            sigma_res = pg.gen_op(sigma_mult[i])
        else:
            sigma_res += sigma_mult[i]

    # last TE/2 nutation
    sigma0 = pg.evolve(sigma_res, Udelay1)

    # instantiate and save transition table of simulation results
    # note. this step copies the TTable1D result from the ACQ into
    #       a TTable1D object in the sim_desc object. Thus, when
    #       we return from this function and the ACQ variable gets
    #       garbage collected, our copy of the results in not affected
    return pg.TTable1D(ACQ.table(sigma0))


def megapress(spin_system, te=68, omega=OMEGA, high_ppm=-7.5, low_ppm=1):
    # ----------------------------------------------------------------
    # This is an example megapress PyGAMMA pulse sequence
    # for use in Vespa-Simulation
    #
    # It expects that the editing and localization pulses
    # have been designed or read in from file via Vespa-RFPulse
    #
    # ----------------------------------------------------------------
    # order of pulse objects from VeSPA
    # so lets load the pulses, and import them as pulse object
    #       - these were exported directly from vespa as pickled objects
    pulse_names = ['siemens_hsinc_400_8750', 'siemens_mao_400_4', 'siemens_rf_gauss_44hz']
    rf_pulses = []

    for ps_name in pulse_names:
        with open('./Simulators/PyGamma/pulses/' + ps_name + '.vps', 'rb') as ps_filename:
            rf_pulses.append(pickle.load(ps_filename))

    # spectrometer frequency in MHz
    specfreq = omega

    # initial evolution time after 90,before 1st 180
    tinitial = float(6.6) / 1000.0  # 6600 for svs_edit

    # echo time
    te = float(te) / 1000.0

    # Pulse centers in ppm for editing pulses
    PulseCenterPPM_local = float(3)  # 3.0
    PulseCenterPPM_edit_on = float(1.9)  # 1.9
    PulseCenterPPM_edit_off = float(7.4)  # 7.5

    bw_pulse_bottom = float(2000)  # 2600.0 - Bandwidth of the pulse at the bottom

    min_peak_hz = (-low_ppm) * specfreq  # 224 Hz at 3T the lowest spin ~1.8 ppm
    max_peak_hz = (-high_ppm) * specfreq  # 570 Hz at 3T the highest frequency spin (~4.6 ppm*spefreq)
    freqoff1 = min_peak_hz  # this is the start of the sweep
    freqoff2 = -(bw_pulse_bottom / 2.0) + min_peak_hz
    freqoff3 = freqoff2
    freqfinal = (bw_pulse_bottom / 2.0) + max_peak_hz

    # number of spatial points (for both x,y)
    numxy_points = int(5)

    # Number of Points in the z,x,y directions for spatial simulation
    Points1 = 5
    Points2 = numxy_points
    Points3 = numxy_points
    Step = (Points2 * specfreq) / (
                freqfinal - freqoff2)  # may make is (freqfinal-freqoff2)/Points2 and remove specfreq from eqn.

    #  Read Pulse Files, Initialize Waveforms --------------------------

    # PulWaveform expects Hz amplitude and angle in degrees, while
    # Simulation gives us mT amplitude and angle in radians.
    # We need to use the appropriate gyromagnetic ratio to do the
    # conversion. This depends on our observe_isotope, since the pulse
    # is not isotope specific, but the spins we expect it to affect,
    # is.
    # (e.g. we use 42576.0 for 1H to covert mT and RAD2DEG for phase)
    obs_iso = '1H'
    gyratio = 42576.0

    # ------  Pulse Setup Section ---------- #
    #
    # - set up RF Pulse containers and read pulse values into them.
    # - format values to units that GAMMA expects
    # - copy into row_vector objects
    # - apply phase roll for off-resonance pulse locations
    # - set up gamma waveform object
    # - set up composite pulse object
    # - get propagator for all steps of the shaped pulse

    # Pulse offsets in  HZ (should be negative or 0)
    B1_offset_local = (0 - PulseCenterPPM_local) * specfreq
    B1_offset_edit_on = (0 - PulseCenterPPM_edit_on) * specfreq
    B1_offset_edit_off = (0 - PulseCenterPPM_edit_off) * specfreq

    # # ------ Excite Pulse
    # vol1, excite_length = pulse2op(rf_pulses[0],
    #                                gyratio,
    #                                "Excite_90",
    #                                spin_system,
    #                                obs_iso,
    #                                offset=B1_offset_local)
    # print(excite_length)

    # # ------ Refocus Pulse
    vol2, refocus_length = pulse2op(rf_pulses[1],
                                    gyratio,
                                    "Refocus_180",
                                    spin_system,
                                    obs_iso,
                                    offset=B1_offset_local)
    vol4 = vol2

    # print(refocus_length)

    # ------ Editing Pulses
    edit1_on, edit1_length = pulse2op(rf_pulses[2],
                                      gyratio,
                                      "siemens_gauss44hz_on",
                                      spin_system,
                                      obs_iso,
                                      offset=B1_offset_edit_on)

    edit1_off, edit1_length = pulse2op(rf_pulses[2],
                                       gyratio,
                                       "siemens_gauss44hz_off",
                                       spin_system,
                                       obs_iso,
                                       offset=B1_offset_edit_off)

    # edit2_off, edit2_on, edit2_length = edit1_off, edit1_on, edit1_length

    edit2_on, edit2_length = pulse2op(rf_pulses[2],
                                      gyratio,
                                      "siemens_gauss44hz_on",
                                      spin_system,
                                      obs_iso,
                                      offset=B1_offset_edit_on)

    edit2_off, edit2_length = pulse2op(rf_pulses[2],
                                       gyratio,
                                       "siemens_gauss44hz_off",
                                       spin_system,
                                       obs_iso,
                                       offset=B1_offset_edit_off)

    # ------ Sequence Timings and GAMMA Initialization ---------- #

    excite_length = 0.0026  # if using Ideal 90y for excite
    # refocus_length = 0.003  # if using ideal 180y refocusing pulses

    # initial evolution time after 90 and before first localization 180
    t1 = tinitial - excite_length / 2 - refocus_length / 2
    te2 = (te - tinitial * 2.0) / 2.0
    t2 = tinitial + te2 - edit1_length - 0.002 - refocus_length
    t3 = 0.002
    t4 = 0.002
    t5 = te2 - refocus_length / 2 - 0.002 - edit2_length

    print ('Megapress pulse seuqence timing:')
    print('     te: ' + str(te))
    print('     Tall: ' + str(t1 + t2 + t3 + t4 + t5 + edit1_length + edit2_length + (refocus_length * 2) + (excite_length / 2.0)))
    print('     t1: ' + str(t1))
    print('     t2: ' + str(t2))
    print('     t3: ' + str(t3))
    print('     t4: ' + str(t4))
    print('     t5: ' + str(t5))


    # set up steady state and observation variables
    H = pg.Hcs(spin_system) + pg.HJ(spin_system)
    D = pg.Fm(spin_system, obs_iso)
    # Set up acquisition
    ac = pg.acquire1D(pg.gen_op(D), H, 0.000001)
    ACQ = ac

    # Calculate delays here (before spatial loop)
    Udelay1 = pg.prop(H, t1)  # First evolution time
    Udelay2 = pg.prop(H, t2)  # Second evolution time
    Udelay3 = pg.prop(H, t3)  # Third evolution time
    Udelay4 = pg.prop(H, t4)  # Fourth evolution time
    Udelay5 = pg.prop(H, t5)  # Fifth evolution time

    # -- Do common first steps outside of spatial loop

    # Equilibrium density matrix
    sigma0 = pg.sigma_eq(spin_system)
    sigma_total = pg.gen_op(sigma0)  # create a copy

    mx_tables = []
    for edit_flag in [False, True]:
        loopcounter = 0
        local_scale = 1.0 / float(Points1 * Points2 * Points3)
        for nss1 in range(Points1):  # slice

            # Apply an ideal 90y pulse
            sigma1 = pg.Ixpuls(spin_system, sigma0, obs_iso, 90.0)

            # Apply a shaped 90 pulse
            # sigma1 = vol1.evolve(sigma0)

            # Evolve for t1
            sigma0 = pg.evolve(sigma1, Udelay1)

            for nss2 in range(Points2):
                for nss3 in range(Points3):
                    # First 180 volume selection pulse - with gradient crushers
                    offsethz2 = freqoff2 + nss2 * specfreq / Step
                    spin_system.offsetShifts(offsethz2)  # note: arg. in Hz
                    # sigma1 = apply_crushed_180_rf(spin_system, sigma0, dephase_ang=[0.0, 90.0], type='crusher')
                    sigma1 = apply_crushed_rf(spin_system, sigma0, vol2, type='crusher')
                    spin_system.offsetShifts(-offsethz2)

                    # Evolve for t2
                    sigma2 = pg.evolve(sigma1, Udelay2)

                    # First BASING bipolar gradient+editing pulse
                    if edit_flag == 0:
                        sigma1 = apply_crushed_rf(spin_system, sigma2, edit1_off, type='bipolar')
                    else:
                        sigma1 = apply_crushed_rf(spin_system, sigma2, edit1_on, type='bipolar')

                    pg.zero_mqc(spin_system, sigma1, 2, 1)  # Keep ZQC and SQC

                    # Evolve for t3
                    sigma2 = pg.evolve(sigma1, Udelay3)

                    # Second 180 volume selection - with gradient crushers
                    offsethz3 = freqoff3 + nss3 * specfreq / Step
                    spin_system.offsetShifts(offsethz3)  # note: arg. in Hz
                    # sigma1 = apply_crushed_180_rf(spin_system, sigma2, type='crusher')
                    sigma1 = apply_crushed_rf(spin_system, sigma2, vol4, type='crusher')
                    spin_system.offsetShifts(-offsethz3)  # note: arg. in Hz

                    # Evolve for t4
                    sigma2 = pg.evolve(sigma1, Udelay4)

                    # Second BASING bipolar gradient+editing pulse
                    if edit_flag == 0:
                        sigma1 = apply_crushed_rf(spin_system, sigma2, edit2_off, type='bipolar')
                    else:
                        sigma1 = apply_crushed_rf(spin_system, sigma2, edit2_on, type='bipolar')

                    pg.zero_mqc(spin_system, sigma1, 2, 1)  # Keep ZQC and SQC

                    # Evolve for t5
                    sigma2 = pg.evolve(sigma1, Udelay5)
                    sigma2 *= local_scale

                    print("System loop index is: " + str(loopcounter) + "  offsethz2 = " + str(
                        offsethz2) + "  offsethz3 = " + str(offsethz3))
                    loopcounter += 1

                    if nss1 + nss2 + nss3 == 0:
                        sigma_total = sigma2
                    else:
                        sigma_total += sigma2
        mx_tables.append(pg.TTable1D(ACQ.table(sigma_total)))
    return mx_tables


def pulse2op(pulse_obj, gyratio, pname, spin_system, obs_iso, offset=0.0):
    """
    A Vespa-Simulation pulse object is passed in via pulse_obj input. The
    gyratio input is a float gyromagnetic ratio, pname is a string name for
    the pulse waveform creates, obs_iso is a string indicating the isotope
    being affected (e.g. '1H') and offset is a float value in Hz for how
    far the complex Vespa pulse object waveform should be shifted from its
    resonance value at 0.0 ppm/Hz.  For example, we would need to shift a
    pulse by 4.7ppm (converted to Hz) in order to center an RF pulse on water
    for a typical simulation.

    PulWaveform expects Hz amplitude and angle in degrees, while
    Simulation gives us mT amplitude and angle in radians.
    We need to use the appropriate gyromagnetic ratio to do the
    conversion. This depends on our observe_isotope, since the pulse
    is not isotope specific, but the spins we expect it to affect,
    is. (e.g. we use 42576.0 for 1H to covert mT and RAD2DEG for phase)

    """
    step = float(pulse_obj['dwell_time']) * 1e-6  # in usec
    wave = pulse_obj['waveform']
    wave = np.array(wave)
    ampl = np.abs(wave) * gyratio
    phas = np.angle(wave) * 180.0 / np.pi  # in radians

    pulse = pg.row_vector(len(wave))
    ptime = pg.row_vector(len(wave))
    for j, val in enumerate(zip(ampl, phas)):
        pulse.put(pg.complex(val[0], val[1]), j)
        ptime.put(pg.complex(step, 0), j)
    plength = pulse.size() * step  # total pulse duration

    if offset != 0.0:
        # typically offset should be 0.0 or negative
        pulse = pg.pulseshift(pulse, ptime, offset)

    pwave = pg.PulWaveform(pulse, ptime, pname)
    pcomp = pg.PulComposite(pwave, spin_system, obs_iso)

    pulse_op = pcomp.GetUsum(-1)

    return pulse_op, plength


def apply_crushed_180_rf(sys, sigma, dephase_ang=[0.0, 90.0, 180.0, 270.0], type='crusher'):
    sigma_mult = []
    for i in dephase_ang:
        sigma_mult.append(pg.gen_op(sigma))

    for i, angle in enumerate(dephase_ang):
        riz = pg.gen_op(pg.Rz(sys, angle))
        sigma_mult[i] = pg.evolve(sigma_mult[i], riz)
        sigma_mult[i] = pg.Iypuls(sys, sigma_mult[i], '1H', 180.0)

        if type == 'bipolar':
            riz = pg.gen_op(pg.Rz(sys, -1 * angle))
        sigma_mult[i] = pg.evolve(sigma_mult[i], riz)
        sigma_mult[i] *= 0.25
        if i == 0:
            sigma_res = pg.gen_op(sigma_mult[i])
        else:
            sigma_res += sigma_mult[i]

    return sigma_res


def apply_crushed_rf(sys, sigma, pulse_op, type='crusher'):
    """
    The sigma input is a single density matrix object.  This object
    is copied into 4 matrices that are rotated about Z axis by the symmetric
    angles in dephase_ang. The pulse_op RF pulse operator is applied to all
    4 density matrices. These are further rotated about the Z axis by the
    symmetric angles in dephase_ang, if type='crusher', or by the negative
    angles in dephase_ang if type='bipolar'. The four matrices are summed
    into one density matrix and divided by 4. A single density matrix is returned.

    after the RF pulse, any spins that did not experience at least a 90 deg RF
    pulse then there will be some signal loss due to lack of refocusing of
    the four matrices.

    """
    dephase_ang = [0.0, 90.0, 180.0, 270.0]

    sigma_mult = []
    for i in dephase_ang:
        sigma_mult.append(pg.gen_op(sigma))

    for i, angle in enumerate(dephase_ang):
        riz = pg.gen_op(pg.Rz(sys, angle))
        sigma_mult[i] = pg.evolve(sigma_mult[i], riz)

        sigma_mult[i] = pulse_op.evolve(sigma_mult[i])

        if type == 'bipolar':
            riz = pg.gen_op(pg.Rz(sys, -1 * angle))
        sigma_mult[i] = pg.evolve(sigma_mult[i], riz)
        sigma_mult[i] *= 0.25
        if i == 0:
            sigma_res = pg.gen_op(sigma_mult[i])
        else:
            sigma_res += sigma_mult[i]

    return sigma_res


def press(spin_system, te1=34, te2=34):
    #----------------------------------------------------------------------
    # This is an example PyGAMMA pulse sequence for use in Vespa-Simulation
    #
    # A timing diagram for this pulse sequence can be found in the Appendix
    # of the Simulation User Manual.
    #----------------------------------------------------------------------

    # the isotope string used to sort/select the metabolites of interest is passed
    # in the sim_desc object so the user can tailor other object within their
    # code to be nuclei specific, such as the observe operator or pulses

    obs_iso = '1H'

    # extract the dynamically changing variable from loop 1 and 2 for 'te1' and
    # 'te2', divide by 1000.0 because the GUI states that values are entered in
    # [ms], but PyGAMMA wants [sec]

    te1 = te1 / 1000.0
    te2 = te2 / 1000.0

    # set up steady state and observation variables
    H   = pg.Hcs(spin_system) + pg.HJ(spin_system)
    D   = pg.Fm(spin_system, obs_iso)
    ac  = pg.acquire1D(pg.gen_op(D), H, 0.000001)
    ACQ = ac
    sigma0 = pg.sigma_eq(spin_system)

    # excite, propagate, refocus and acquire the data
    sigma1 = pg.Iypuls(spin_system, sigma0, obs_iso, 90.0)
    Udelay = pg.prop(H, te1*0.5)
    sigma0 = pg.evolve(sigma1, Udelay)
    sigma1 = pg.Iypuls(spin_system, sigma0, obs_iso, 180.0)
    Udelay = pg.prop(H, (te1+te2)*0.5)
    sigma0 = pg.evolve(sigma1, Udelay)
    sigma1 = pg.Iypuls(spin_system, sigma0, obs_iso, 180.0)
    Udelay = pg.prop(H, te2*0.5)
    sigma0 = pg.evolve(sigma1, Udelay)

    # instantiate and save transition table of simulation results
    # note. this step copies the TTable1D result from the ACQ into
    #       a TTable1D object in the sim_desc object. Thus, when
    #       we return from this function and the ACQ variable gets
    #       garbage collected, our copy of the results in not affected
    return pg.TTable1D(ACQ.table(sigma0))
