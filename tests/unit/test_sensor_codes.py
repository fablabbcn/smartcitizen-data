from scdata.device.process.params import alphasense_sensor_codes, as_sensor_algs


def test_a_and_b_series_have_their_own_codes():
    assert alphasense_sensor_codes['133'] == 'ASA4_H2S' and alphasense_sensor_codes['163'] == 'ASB4_H2S'
    assert alphasense_sensor_codes['130'] == 'ASA4_NO' and alphasense_sensor_codes['160'] == 'ASB4_NO'
    assert len(set(alphasense_sensor_codes.values())) == len(alphasense_sensor_codes)
    # Every code has its temperature compensation
    assert set(alphasense_sensor_codes.values()) <= set(as_sensor_algs)
