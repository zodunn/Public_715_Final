

class Ray:
    def __init__(self, e, t, d):
        self.e = e
        self.t = t
        self.d = d

    def collide(self, vert_a, vert_b, vert_c):
        a = vert_a[0] - vert_b[0]
        b = vert_a[1] - vert_b[1]
        c = vert_a[2] - vert_b[2]

        d = vert_a[0] - vert_c[0]
        e = vert_a[1] - vert_c[1]
        f = vert_a[2] - vert_c[2]

        g = self.d[0]
        h = self.d[1]
        i = self.d[2]

        j = vert_a[0] - self.e[0]
        k = vert_a[1] - self.e[1]
        l = vert_a[2] - self.e[2]

        M = a * (e*i - h*f) + b * (g*f - d*i) + c * (d*h - e*g)
        if M == 0:
            return False, 0

        t = -(f * (a*k - j*b) + e * (j*c - a*l) + d * (b*l - k*c)) / M

        if t > 0:
            gamma = (i * (a*k - j*b) + h * (j*c - a*l) + g * (b*l - k*c)) / M
            if gamma > 0:
                beta = (j * (e*i - h*f) + k * (g*f - d*i) + l * (d*h - e*g)) / M
                if beta > 0:
                    if beta + gamma < 1:
                        return True, t
        return False, t
