import math

def get_new_coordinates(ix, iy, jx, jy):
    """Gets new set of coordinates closer to each other"""
    # Compute the Euclidean distance between the original points
    d = math.sqrt((ix - jx) ** 2 + (iy - jy) ** 2)
    if d == 0:
        return ix, iy, jx, jy

    # Compute the midpoint of the line segment connecting the original points
    mx = (ix + jx) / 2
    my = (iy + jy) / 2

    # Compute the unit vector in the direction of the line segment connecting the original points
    ux = (jx - ix) / d
    uy = (jy - iy) / d
    print(ux, uy)

    # Compute the perpendicular unit vector to the line segment
    vx = -uy
    vy = -ux

    # Compute the coordinates of the new points
    #k = d / 2  # distance from midpoint to new points
    k = math.sqrt(d ** 2 - (d / 2) ** 2)  # distance from midpoint to new points
    new_ix = round(ix + ux)
    new_iy = round(iy + uy)
    new_jx = round(jx - ux)
    new_jy = round(jy - uy)
    return (new_ix, new_iy, new_jx, new_jy)

if __name__ == '__main__':
    # Define the coordinates of the original points
    assert get_new_coordinates(1, 1, 1, 1) == (1, 1, 1, 1)
    assert get_new_coordinates(1, 1, 5, 5) == (2, 2, 4, 4)
    assert get_new_coordinates(5, 5, 1, 1) == (4, 4, 2, 2)
    assert get_new_coordinates(-5, -5, 5, 5) == (-4, -4, 4, 4)
    assert get_new_coordinates(1, 1, 1, 2) == (1, 2, 1, 1)
    ix, iy = 5, 5
    jx, jy = 1, 1
    new_ix, new_iy, new_jx, new_jy = get_new_coordinates(ix, iy, jx,  jy)
    print(f"Original points: ({ix},{iy}), ({jx},{jy})")
    print(f"New points: ({new_ix},{new_iy}), ({new_jx},{new_jy})")
