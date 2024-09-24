import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from draw_tdcr import draw_tdcr
import copy
from utils import setupfigure


class Section:
    """
    Calculates the configuration of a section of a tendon driven continuum robot based on constant curvature model
    given the data regarding the length, curvature, angle of twist, and the number of disks.
    """

    def __init__(self, n, l, kappa, phi, base_tf, kappa_range=None, phi_range=None):
        """
        Creates a section

        :param n: The number of disks in the section
        :type n: int
        :param l: Section length
        :type l: float
        :param kappa: Section curvature
        :type kappa: float
        :param phi: The angle of twist
        :type phi: float
        :param base_tf: The transfer function of the base of the section
        :type base_tf: ndarray
        :param kappa_range: Curvature range of the section, used to calculate the workspace
        :type kappa_range: tuple, optional
        :param phi_range: Angle of twist range of the section, used to calculate the workspace
        :type phi_range: tuple, optional
        """
        self._base_tf = base_tf
        self.n = n
        self.kappa = kappa
        self.l = l
        self.phi = phi
        self.kappa_range = kappa_range
        self.phi_range = phi_range

    @property
    def g(self):
        """
        Get the transformation matrix for all disks in the section

        The 4 by 4 transformation matrices are reshaped into 1 by 16 vector and packed into an n by 16 numpy array
        """
        self._configuration()
        return self._g

    @property
    def end_tf(self):
        """
        Get the transformation matrix of the tip of the section
        """
        self._configuration()
        return self._end_tf

    @property
    def base_tf(self):
        """
        Get the transformation matrix of the base of the section
        """
        return self._base_tf

    @base_tf.setter
    def base_tf(self, base_tf):
        self._base_tf = base_tf

    def _configuration(self):
        """
        Calculates the transformation matrix of each disk and provides the configuration
        of the section by setting the g matrix
        """
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

    def _position(self, s, kappa, phi):
        """
        Calculates the position vector of a disk based on constant curvature model

        Accounts for the singularity in the model when the curvature is zero

        :param s: Distance of the disk from the base along the backbone
        :type s: float
        :param kappa: Section curvature
        :type kappa: float
        :param phi: The angle of twist
        :type phi: float
        """
        if self.kappa == 0:
            return np.array([[0], [0], [s]])

        return np.array(
            [
                [(np.cos(phi) / kappa) * (1 - np.cos(kappa * s))],
                [(np.sin(phi) / kappa) * (1 - np.cos(kappa * s))],
                [np.sin(kappa * s) / kappa],
            ]
        )

    @staticmethod
    def R_z(phi):
        """
        Returns the rotation matrix around z axis given an angle of phi

        :param phi: The angle of rotation
        :type phi: float
        """
        return np.array(
            [[np.cos(phi), -np.sin(phi), 0], [np.sin(phi), np.cos(phi), 0], [0, 0, 1]]
        )

    @staticmethod
    def R_y(theta):
        """
        Returns the rotation matrix around y axis given an angle of theta

        :param theta: The angle of rotation ("kappa*s" in the CC model)
        :type theta: float
        """
        return np.array(
            [
                [np.cos(theta), 0, np.sin(theta)],
                [0, 1, 0],
                [-np.sin(theta), 0, np.cos(theta)],
            ]
        )


class TDCR_PCC:
    """
    Creates a tendon driven continuum robot object based on constant curvature model
    """

    def __init__(self):
        """
        Initializes a tdcr object
        """
        self._sections = []

    def __str__(self):
        """
        Provides details regarding the specifications of the sections of the robot as the string output of the object
        """
        section_details = []
        for i, section in enumerate(self.sections):
            if i == 0:
                section_details.append(
                    f"Section {i+1}: Length = {section.l:.3f}, Curvature = {section.kappa:.3f}, Phi = {section.phi:.3f}, Number of Disks = {section.n}"
                )
            else:
                section_details.append(
                    f"Section {i+1}: Length = {section.l:.3f}, Curvature = {section.kappa:.3f}, Phi = {section.phi:.3f}, Number of Disks = {section.n-1}"
                )

        return "\n".join(section_details)

    @property
    def g(self):
        """
        Get the transformation matrix for all disks in the section

        The 4 by 4 transformation matrices are reshaped into 1 by 16 vector and packed into an n by 16 numpy array

        :type: ndarray
        """
        self.calculate_g()
        return self._g

    @property
    def sections(self):
        """
        A list of sections of type Seciton

        :type: list
        """
        return self._sections

    @property
    def end_tf(self):
        """
        Get the transformation matrix of the tip of the robot

        :type: ndarray
        """
        return self.g[-1].reshape((4, 4), order="F")

    @property
    def section_ends(self):
        """
        Get the disks in which the sections end
        """
        i = self.sections[0].n
        section_ends = [i]
        for section in self.sections[1:]:
            i += section.n - 1
            section_ends.append(i)
        return section_ends

    def add_section(self, n, l, kappa, phi=0):
        """
        Adds a section to the robot

        :param n: The number of disks in the section
        :type n: int
        :param l: Section length
        :type l: float
        :param kappa: Section curvature
        :type kappa: float
        :param phi: The angle of twist
        :type phi: float, optional
        """
        if len(self.sections) == 0:
            base_tf = np.eye(4)
            new_section = Section(n, l, kappa, phi, base_tf)
            self._sections.append(new_section)
        else:
            base_tf = self.sections[-1].end_tf
            new_section = Section(n + 1, l, kappa, phi, base_tf)
            self._sections.append(new_section)

    def modify_section(self, index, n=None, l=None, kappa=None, phi=None):
        """
        Modifies a section of the robot and modifies the following sections to match this variation.
        Re-calculates g in the end. Takes four positional parameters. No modifications are made if the parameters are not provided.

        :param index: The section that is being modified (0-indexed)
        :type index: int
        :param n: The number of disks in the section
        :type n: int, optional
        :param l: Section length
        :type l: float, optional
        :param kappa: Section curvature
        :type kappa: float, optional
        :param phi: The angle of twist
        :type phi: float, optional
        """
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
        """
        Calculates and sets the g array
        """
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
        """
        Visualizes the robot using draw_tdcr.py module. Takes similar set of parameters
        """
        if len(self.sections) == 0:
            raise ValueError("The robot has no sections")
        draw_tdcr(self.g, np.array(self.section_ends), **kwargs)

    def get_disk_tf(self, i):
        """
        Calculates the transfer function of a given disk

        :param i: The disk number
        :type i: int
        """
        return self.g[i].reshape((4, 4), order="F")

    def workspace(self, ranges=None, kappa_step=0.5, phi_step=0.5, ax=None, plot_midpoints=False):
        """
        Visualizes the workspace of the robot.
        Shows the plot only if the ax parameter is not provided.

        :param ranges: A list of dictionaries, each corresponding to a section having the keys of "kappa_range" and "phi_range".
                       The values are tuples of size 2 indicating the range of kappa and phi.
        :type ranges: list, optional (if provided during section definition)
        :param kappa_step: Increment of kappa for workspace calculation
        :type kappa_step: float, optional
        :param phi_step: Increment of phi for workspace calculation
        :type phi_step: float, optional
        :param plot_midpoints: If True, plots all of the disks rather than the end effector
        :type plot_midpoints: boolean, optional
        """
        if ranges == None:
            ranges = []
            for i, section in enumerate(self.sections):
                if section.kappa_range == None:
                    raise ValueError(f"Please specify a kappa range for section {i+1}")
                elif section.phi_range == None:
                    raise ValueError(f"Please specify a phi range for section {i+1}")
                else:
                    ranges.append(
                        {
                            "kappa_range": section.kappa_range,
                            "phi_range": section.phi_range,
                        }
                    )
        if len(ranges) != len(self.sections):
            raise ValueError("The number of ranges must match the number of sections")

        showplot = False
        if ax == None:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d")
            showplot = True
        obj = copy.deepcopy(self)
        kappa_range = [rng["kappa_range"] for rng in ranges]
        phi_range = [rng["phi_range"] for rng in ranges]

        self._compute_workspace(obj, ax, kappa_range, phi_range, kappa_step, phi_step, plot_midpoints)

        if showplot == True:
            plt.show()

    @staticmethod
    def _compute_workspace(robot, ax, kappa_range, phi_range, kappa_step, phi_step, plot_midpoints):
        """
        Generates dynamic nested for loops to populate the plot with the workspace points

        :param robot: The robot for which the workspace is calculated
        :type robot: TDCR_PCC object
        :param ax: Matplotlib subplot
        :param kappa_range: Ranges of curvature for the sections
        :type kappa_range: list of tuples
        :param phi_range: Ranges of phi for the sections
        :type phi_range: list of tuples
        :param kappa_step: Increment of kappa for workspace calculation
        :type kappa_step: float, optional
        :param phi_step: Increment of phi for workspace calculation
        :type phi_step: float, optional
        :param plot_midpoints: If True, plots all of the disks rather than the end effector
        :type plot_midpoints: boolean, optional
        """
        
        def recursive(k, kappa, phi):
            i = len(robot.sections) - k - 1

            #Plot scattering at the end of the recursion
            if k == 0:
                if plot_midpoints == True:
                    cmap = plt.get_cmap('twilight')
                    for z, section in enumerate(robot.sections):
                        color = cmap(z/len(robot.sections))
                        for g in section.g:
                            x, y, z = g[12:15]
                            ax.scatter(x, y, z, color = color)
                else:
                    x, y, z = robot.end_tf[:3, 3]
                    ax.scatter(x, y, z, color="k")
                return

            #Recursion
            if kappa_range[i][0] == kappa_range[i][1]:
                recursive(k-1, kappa_range[i][0], phi_range[i][0])
            else:
                for kp in np.arange(kappa_range[i][0], kappa_range[i][1] + kappa_step/10, kappa_step):
                    if phi_range[i][0] == phi_range[i][1]:
                        robot.modify_section(i, kappa=kp)
                        recursive(k - 1, kp, phi_range[i][0])
                    else:
                        for ph in np.arange(phi_range[i][0], phi_range[i][1] + phi_step/10, phi_step):
                            robot.modify_section(i, kappa=kp, phi=ph)
                            recursive(k - 1, kp, ph)

        recursive(len(robot.sections) - 1, 0, 0)


def main():
    # Instantiating robot object
    robot = TDCR_PCC()

    # Section properties
    diameter = 6.2e-3
    l1 = 100e-3
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
    # robot.modify_section(0, l=50e-3, kappa=0)

    # Results
    print("Robot specifications: \n", robot, sep="")
    print("\nThe transfer function for the tip: \n", robot.end_tf)

    x = 5
    print(f"\nThe transfer function for the {x}th disk is: \n", robot.get_disk_tf(x))

    ax = setupfigure(
        robot.g,
        robot.section_ends,
        tipframe=True,
        segframe=False,
        baseframe=False,
        projections=False,
        baseplate=True,
    )
    ax.view_init(elev=0, azim=90)

    ranges = [
        {"kappa_range": (-20, 20), "phi_range": (0, 0)},
        {"kappa_range": (-10, 10), "phi_range": (0, 0)},
        {"kappa_range": (-10, 10), "phi_range": (0, 0)},
    ]

    robot.workspace(ranges=ranges, kappa_step=4, phi_step=0.2, ax=ax, plot_midpoints=True)
    robot.draw(r_disk=diameter / 2, ax=ax)

    plt.show()


if __name__ == "__main__":
    main()
