### Script to preprocess data before feeding the CNN : Scaling, Padding, outlier management...

def MinMaxScaling(signal):
    """Scales the XRD pattern between 0 and 1 to have the same treatment between data
        Args:
            - signal (list) : list of floats corresponding to the intensity
        Returns:
            A scaled list of floats between 0 and 1
    """
    min_signal = min(signal)
    max_signal = max(signal)
    if abs(min_signal - max_signal) < 1e-3:
        raise Exception("Difference between min and max is close from zero")
    else:
        return [(x - min_signal) / (max_signal - min_signal) for x in signal]


def performPadding(pattern):
    """Pads the intensity with zeros where the range of the signal is not defined
    We get a new signal with angles between 5 and 85 degrees
        Args:
            - pattern (tuple) : tuple of two lists, angles and intensities
        Returns:
            A new pattern with padded angles and intensities"""
    angles, intensities = pattern
    new_angles, new_intensities = [], []
    step = angles[1] - angles[0]        # Step between two angles
    # Build the new lists
    min_angle = angles[0]

    # We pad on the left with zeros
    if min_angle > 5:
        new_angles.append(5)
        new_intensities.append(0)
        while new_angles[-1] < min_angle:
            new_angles.append(new_angles[-1] + step)
            new_intensities.append(0)
    
    # We concat with the intensities and angles from the unpadded pattern
    new_angles = new_angles + angles
    new_intensities = new_intensities + intensities

    # We pad on the right with zeros
    max_angle = angles[-1]
    if max_angle < 85:
        while new_angles[-1] < 85:
            new_angles.append(new_angles[-1] + step)
            new_intensities.append(0)
    
    return (new_angles, new_intensities)