import numpy as np
from draw_tdcr import draw_tdcr


class Section:
    """ """

    def __init__(self, n, l, kappa, phi, base_tf):
        self._base_tf = base_tf
        self.n = n
        self.kappa = kappa
        self.l = l
        self.phi = phi

    @property
    def g(self):
        self._configuration()
        return self._g

    @property
    def end_tf(self):
        self._configuration()
        return self._end_tf

    @property
    def base_tf(self):
        return self._base_tf

    @base_tf.setter
    def base_tf(self, base_tf):
        self._base_tf = base_tf

    def _configuration(self):
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
                self._end_tf = transformation
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


class TDCR_PCC:
    """ """

    def __init__(self):
        self._sections = []

    def __str__(self):
        section_details = []
        for i, section in enumerate(self.sections):
            if i == 0:
                section_details.append(
                    f"Section {i}: Length = {section.l:.3f}, Curvature = {section.kappa:.3f}, Phi = {section.phi:.3f}, Number of Disks = {section.n}"
                )
            else:
                section_details.append(
                    f"Section {i}: Length = {section.l:.3f}, Curvature = {section.kappa:.3f}, Phi = {section.phi:.3f}, Number of Disks = {section.n-1}"
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

    def modify_section(self, index, n=None, l=None, kappa=None, phi=None):
        if n is not None:
            self._sections[index].n = n
        if l is not None:
            self._sections[index].l = l
        if kappa is not None:
            self._sections[index].kappa = kappa
        if phi is not None:
            self._sections[index].phi = phi
        for i, section in enumerate(self.sections[index + 1 :]):
            section.base_tf = self.sections[index + i].end_tf
        self.calculate_g()

    def calculate_g(self):
        if len(self.sections) == 0:
            raise ValueError("The robot has no secitons")
        g_sections = []
        for i, section in enumerate(self.sections):
            if i == 0:
                g_sections.append(section.g)
            else:
                g_sections.append(section.g[1:])

        self._g = np.concatenate(g_sections, axis=0)

    def draw(self, **kwargs):
        if len(self.sections) == 0:
            raise ValueError("The robot has no sections")
        i = self.sections[0].n
        section_ends = [i]
        for section in self.sections[1:]:
            i += section.n - 1
            section_ends.append(i)

        draw_tdcr(self.g, np.array(section_ends), **kwargs)

    def get_disk_tf(self, i):
        self.calculate_g()
        return self.g[i].reshape((4, 4), order='F')


def main():
    # Instantiating robot object
    robot = TDCR_PCC()

    # Section properties
    diameter = 6.2e-3
    l1 = 200e-3
    l2 = 50e-3
    l3 = 50e-3
    rho1 = 2 * l1 / np.pi
    kappa1 = 1 / rho1
    kappa2 = 0
    kappa3 = 0

    # Adding sections
    robot.add_section(20, l1, kappa1)
    robot.add_section(5, l2, kappa2)
    robot.add_section(5, l3, kappa3)

    # Section modifiction
    robot.modify_section(0, l=50e-3, kappa=0)

    # Results
    print("Robot specifications: \n", robot, sep="")
    print("\nThe transfer function for the tip: \n", robot.end_tf)

    x = 5
    print(f"\nThe transfer function for the {x}th disk is: \n", robot.get_disk_tf(x))

    robot.draw(r_disk=diameter / 2)


if __name__ == "__main__":
    main()
