import numpy as np
from draw_tdcr import draw_tdcr

class Section():
    def __init__(self, n, l, kappa, phi, base_tf):
        self.base_tf = base_tf
        self.n = n
        self.kappa = kappa
        self.l = l
        self.phi = phi
        self.g = self.configuration()
    
    
    def configuration(self):
        g = np.zeros((self.n,16))
        points = np.linspace(0, self.l, self.n)
        for i, s in enumerate(points):
            pos = np.concatenate((self.position(s, self.kappa, self.phi), np.array([[1]])), axis= 0)
            rotation = np.concatenate((self.R_z(self.phi) @ self.R_y(self.kappa*s) @ self.R_z(-self.phi), np.array([[0,0,0]])), axis= 0)
            transformation = self.base_tf @ np.concatenate((rotation, pos), axis= 1)
            if i == (self.n-1):
                self.end_tf = transformation
            g[i] = transformation.flatten(order= 'F')

        return g

    @staticmethod
    def R_z(phi):
        return np.array([
            [np.cos(phi), -np.sin(phi), 0],
            [np.sin(phi), np.cos(phi), 0],
            [0, 0, 1]
        ])

    @staticmethod
    def R_y(theta):
        return np.array([
            [np.cos(theta), 0, np.sin(theta)],
            [0, 1, 0],
            [-np.sin(theta), 0, np.cos(theta)]
            ])

    @staticmethod
    def position(s, kappa, phi):
        return np.array([
                [(np.cos(phi)/kappa)*(1-np.cos(kappa*s))], 
                [(np.sin(phi)/kappa)*(1-np.cos(kappa*s))], 
                [np.sin(kappa*s)/kappa]
                ])


class TDCR():
    def __init__(self):
        self.sections = []
        self.section_number = 0

    def add_section(self, n, l, kappa, phi):
        if len(self.sections) == 0:
            base_tf = np.eye(4)
        else:
            base_tf = self.sections[-1].end_tf      
        self.sections.append(Section(n, l, kappa, phi, base_tf))
        g_sections = []
        for section in self.sections:
            g_sections.append(section.g)
        self.g = np.concatenate(g_sections, axis= 0)

    def draw(self):
        section_ends = []
        i = 0
        for section in self.sections:
            i += section.n
            section_ends.append(i)
        # print(section_ends)
        draw_tdcr(self.g, np.array(section_ends))



    # def end_tf(self):
    #     return (self.sections[-1].end_tf)

def main():
    robot = TDCR()

    robot.add_section(10, 100e-3, 1/100e-3, 1)
    robot.add_section(10, 200e-3, 0.01, 1)
    robot.draw()
    # print(robot.g)



if __name__ == "__main__":
    main()
