import numpy as np
from draw_tdcr import draw_tdcr


class Section:
    def __init__(self, n, l, kappa, phi, base_tf):
        self.base_tf = base_tf
        self.n = n
        self.kappa = kappa
        self.l = l
        self.phi = phi
        self.configuration()

    @property
    def g(self):
        return self._g

    def configuration(self):
        g = np.zeros((self.n, 16))
        points = np.linspace(0, self.l, self.n)
        for i, s in enumerate(points):
            pos = np.concatenate(
                (self._position(s, self.kappa, self.phi), np.array([[1]])), axis=0
            )
            rotation = np.concatenate(
                (
                    self.R_z(self.phi) @ self.R_y(self.kappa * s) @ self.R_z(-self.phi),
                    np.array([[0, 0, 0]]),
                ),
                axis=0,
            )
            transformation = self.base_tf @ np.concatenate((rotation, pos), axis=1)
            if i == (self.n - 1):
                self.end_tf = transformation
            g[i] = transformation.flatten(order="F")
        self._g = g

    @staticmethod
    def R_z(phi):
        return np.array(
            [[np.cos(phi), -np.sin(phi), 0], [np.sin(phi), np.cos(phi), 0], [0, 0, 1]]
        )

    @staticmethod
    def R_y(theta):
        return np.array(
            [
                [np.cos(theta), 0, np.sin(theta)],
                [0, 1, 0],
                [-np.sin(theta), 0, np.cos(theta)],
            ]
        )

    def _position(self, s, kappa, phi):
        if self.kappa == 0:
            return np.array([[0], [0], [s]])

        return np.array(
            [
                [(np.cos(phi) / kappa) * (1 - np.cos(kappa * s))],
                [(np.sin(phi) / kappa) * (1 - np.cos(kappa * s))],
                [np.sin(kappa * s) / kappa],
            ]
        )


class TDCR:
    def __init__(self):
        self._sections = []

    def __str__(self):
        section_details = []
        for i, section in enumerate(self.sections):
            section_details.append(
                f"Section {i}: Length = {section.l:.3f}, Curvature = {section.kappa:.3f}, Phi = {section.phi:.3f}, Number of Disks = {section.n}"
            )

        return "\n".join(section_details)

    @property
    def g(self):
        self.calculate_g()
        return self._g

    @property
    def sections(self):
        return self._sections

    @property
    def end_tf(self):
        return self._end_tf

    def add_section(self, n, l, kappa, phi=0):
        if len(self.sections) == 0:
            base_tf = np.eye(4)
            new_section = Section(n, l, kappa, phi, base_tf)
            self._sections.append(new_section)
        else:
            base_tf = self.sections[-1].end_tf
            new_section = Section(n + 1, l, kappa, phi, base_tf)
            self._sections.append(new_section)

        self._end_tf = self.sections[-1].end_tf

    def modify_section(i, n, l, kappa, phi):
        self._sections[i] = Section(n, l, kappa, phi=0)

    def calculate_g(self):
        if len(self.sections) == 0:
            raise ValueError("The robot has no secitons")
            # self._g = None
        g_sections = []
        for i, section in enumerate(self.sections):
            if i == 0:
                g_sections.append(section.g)
            else:
                g_sections.append(section.g[1:])

        self._g = np.concatenate(g_sections, axis=0)

    def draw(self):
        if len(self.sections) == 0:
            raise ValueError("The robot has no sections")
        i = self.sections[0].n
        section_ends = [i]
        for section in self.sections[1:]:
            i += section.n - 1
            section_ends.append(i)

        print(section_ends)
        draw_tdcr(self.g, np.array(section_ends))


def main():
    robot = TDCR()
    l1 = 200e-3
    rho1 = 2 * l1 / np.pi
    robot.add_section(10, l1, 1 / rho1)
    robot.add_section(5, 50e-3, 0)
    robot.add_section(5, 50e-3, 0)

    # print(robot.end_tf)
    print(robot)
    robot.draw()

    # draw_tdcr(robot.g, robot.draw())

    # print(robot.g)


if __name__ == "__main__":
    main()
