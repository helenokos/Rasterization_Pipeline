from PyQt5 import QtGui, QtCore, QtWidgets
from PyQt5.QtCore import Qt
from GUI import Ui_MainWindow
import sys
import math
import numpy as np
import matplotlib.pyplot as plt

class obj_info():
    def __init__(self):
        self.offset = [[-1,-1,-1], [1,-1,-1], [-1,1,-1], [1,1,-1], [0,0,1]]
        self.c = point()
        self.points = [[0,0,0], [2,0,0], [0,2,0], [2,2,0], [1,1,2]]
        self.lines = [[1,2,4], [3,4], [3,4], [4]]
        self.is_inside = [True, True, True, True, True]
        self.all_edges = []
        self.color = ['g', 'b', 'c', 'y', 'm']

class frustum():
    def __init__(self):
        self.edge = ['l', 'r', 't', 'b', 'n', 'f']
        self.l = self.r = self.t = self.b = self.n = self.f = 0

class point():
    def __init__(self):
        self.x = self.y = self.z = 0

class main_window(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        #combo box
        choice = ['xy','yz','xz']
        self.ui.re_c.addItems(choice)

        #click
        self.ui.sc_btn.clicked.connect(self.cal_sc)
        self.ui.ro_btn.clicked.connect(self.cal_ro)
        self.ui.re_btn.clicked.connect(self.cal_re)
        self.ui.tr_btn.clicked.connect(self.cal_tr)
        self.ui.btn_world.clicked.connect(self.world)
        self.ui.btn_view.clicked.connect(self.view)
        self.ui.btn_clip.clicked.connect(self.clip)
        self.ui.btn_normal.clicked.connect(self.norm)
        self.ui.btn_image.clicked.connect(self.image)

    #scaling
    def cal_sc(self):
        x = float(self.ui.sc_x.toPlainText())
        y = float(self.ui.sc_y.toPlainText())
        z = float(self.ui.sc_z.toPlainText())

        for idx, point in enumerate(self.obj.points):
            self.obj.points[idx] = [point[0]*x, point[1]*y, point[2]*y]
        
        self.draw_3d_graph()
    
    #rotation
    def cal_ro(self):
        theta = float(self.ui.ro.toPlainText())*math.pi
        cos = np.around(math.cos(theta), decimals=5)
        sin = np.around(math.sin(theta), decimals=5)
        ro_matrix = np.matrix([[cos, -sin, 0],[sin, cos, 0], [0, 0, 1]])
        xy_matrix = np.matrix(self.obj.points).T
        self.obj.points = np.array((ro_matrix.dot(xy_matrix)).T)

        self.draw_3d_graph()
   
    #reflection
    def cal_re(self):
        xy_matrix = np.matrix(self.obj.points).T
        if self.ui.re_c.currentIndex() == 0:
            re_matrix = np.matrix([[1, 0, 0], [0, 1, 0], [0, 0, -1]])
        elif self.ui.re_c.currentIndex() == 1:
            re_matrix = np.matrix([[-1, 0, 0], [0, 1, 0], [0, 0, 1]])
        else:
            re_matrix = np.matrix([[1, 0, 0], [0, -1, 0], [0, 0, 1]])
        self.obj.points = np.array((re_matrix.dot(xy_matrix)).T)
        self.draw_3d_graph()
    
    #translation
    def cal_tr(self):
        a = float(self.ui.tr_x.toPlainText())
        b = float(self.ui.tr_y.toPlainText())
        c = float(self.ui.tr_z.toPlainText())
        for idx, point in enumerate(self.obj.points):
            x, y, z = point
            self.obj.points[idx] = [x+a, y+b, z+c]
        self.draw_3d_graph()
    
    def my_reset(self):
        self.obj = obj_info()
        self.camera = point()
        self.coordinate = []
        self.cam_fru = frustum()
    
    def world(self):
        self.my_reset()

        # get camera position
        self.camera.x = float(self.ui.camera_x.text())
        self.camera.y = float(self.ui.camera_y.text())
        self.camera.z = float(self.ui.camera_z.text())

        #get object center
        self.obj.c.x = float(self.ui.obj_x.text())
        self.obj.c.y = float(self.ui.obj_y.text())
        self.obj.c.z = float(self.ui.obj_z.text())

        #set object points
        self.obj.points = [np.array([self.obj.c.x, self.obj.c.y, self.obj.c.z])+np.array(i) for i in self.obj.offset]

        self.draw_3d_graph()
        self.show_obj_points_in_UI()

    def view(self):
        #set camera to origin and compute obj points
        self.obj.c.x = self.obj.c.x - self.camera.x
        self.obj.c.y = self.obj.c.y - self.camera.y
        self.obj.c.z = self.obj.c.z - self.camera.z
        self.obj.points = [np.array([x-self.camera.x, y-self.camera.y, z-self.camera.z]) for x,y,z in self.obj.points]
        self.camera.x = self.camera.y = self.camera.z = 0

        #vector
        vector = [self.obj.c.x, self.obj.c.y, self.obj.c.z]

        #new_x coordinate
        new_x = np.cross(vector, [0,1,0])
        new_x = new_x/np.linalg.norm(new_x)
        self.ui.new_x.setText(self.np_array_to_string(new_x))

        #new_y coordinate
        new_y = np.cross(new_x, vector)
        new_y = new_y/np.linalg.norm(new_y)
        self.ui.new_y.setText(self.np_array_to_string(new_y))

        #new_z coordinate
        new_z = np.array([-i for i in vector])
        new_z = new_z/np.linalg.norm(new_z)
        self.ui.new_z.setText(self.np_array_to_string(new_z))

        #store coordinate information
        self.coordinate = [new_x, new_y, new_z]        
        
        #compute translated point
        matrix = np.matrix([[j[i] for j in self.coordinate]for i in range(3)])
        self.obj.points = np.array(((matrix.I).dot(np.matrix(self.obj.points).T).T))

        self.draw_3d_graph()
        self.show_obj_points_in_UI()

    def clip(self):
        self.mapping()
        self.clipping()
        self.draw_3d_graph(is_norm=True)
    
    def mapping(self):
        #loading frustum info
        for line in self.cam_fru.edge:
            command = 'self.cam_fru.{} = float(self.ui.camera_{}.text())'.format(line, line)
            exec(command)

        #build mapping matrix
        matrix = np.matrix([[2/(self.cam_fru.r-self.cam_fru.l), 0, 0, (self.cam_fru.l+self.cam_fru.r)/(self.cam_fru.l-self.cam_fru.r)],[0, 2/(self.cam_fru.t-self.cam_fru.b), 0, (self.cam_fru.b+self.cam_fru.t)/(self.cam_fru.b-self.cam_fru.t)],[0,0,2/(self.cam_fru.n-self.cam_fru.f), (self.cam_fru.f+self.cam_fru.n)/(self.cam_fru.f-self.cam_fru.n)], [0,0,0,1]])

        #mapping obj to frustum
        self.obj.points = [np.append(i, 1) for i in self.obj.points]
        self.obj.points = np.array(matrix.dot(np.matrix(self.obj.points).T).T)
        
    def clipping(self):
        #check if the point is in frustum
        for idx, point in enumerate(self.obj.points):
            is_inside = False
            if point[0] >= -1 and point[0] <= 1:
                if point[1] >= -1 and point[1] <= 1:
                    if point[2] >= -1 and point[2] <= 1:
                        is_inside = True
                        command = 'self.ui.Nobj_{}.setText("{}")'.format(idx+1,self.np_array_to_string(point))
                        exec(command)
            self.obj.is_inside[idx] = is_inside
            if not is_inside:
                command = 'self.ui.Nobj_{}.setText("outside")'.format(idx+1)
                exec(command)

        #clipping points
        for i in range(4):
            for j in self.obj.lines[i]:
                if self.obj.is_inside[i] and self.obj.is_inside[j]:
                    self.obj.all_edges.append([self.obj.points[i], self.obj.points[j]])
                else:
                    vector = f = t = 0
                    if (self.obj.is_inside[i]) and (not self.obj.is_inside[j]):
                        vector = self.obj.points[j]-self.obj.points[i]
                        f = self.obj.points[i]
                    elif (not self.obj.is_inside[i]) and (self.obj.is_inside[j]):
                        vector = self.obj.points[i]-self.obj.points[j]
                        f = self.obj.points[j]
                    else:
                        continue
                    done = False
                    
                    #l
                    if vector[0] != 0:
                        c = (-1-f[0])/vector[0]
                        if c > 0:
                            y = f[1]+c*vector[1]
                            z = f[2]+c*vector[2]
                            t = np.array([-1, y, z])
                            if (y >= -1 and y <= 1 and z >= -1 and z <= 1):
                                t = np.array([-1, y, z])
                                done = True
                    #r
                    if not done and vector[0] != 0:
                        c = (1-f[0])/vector[0]
                        if c > 0:
                            y = f[1]+c*vector[1]
                            z = f[2]+c*vector[2]
                            if (y >= -1 and y <= 1 and z >= -1 and z <= 1):
                                t = np.array([1, y, z])
                                done = True
                    #b
                    if not done and vector[1] != 0:
                        c = (-1-f[1])/vector[1]
                        if c > 0:
                            x = f[0]+c*vector[0]
                            z = f[2]+c*vector[2]
                            if (x >= -1 and x <= 1 and z >= -1 and z <= 1):
                                t = np.array([x, -1, z])
                                done = True
                    #t
                    if not done and vector[1] != 0:
                        c = (1-f[1])/vector[1]
                        if c > 0:
                            x = f[0]+c*vector[0]
                            z = f[2]+c*vector[2]
                            if (x >= -1 and x <= 1 and z >= -1 and z <= 1):
                                t = np.array([x, 1, z])
                                done = True
                    #n
                    if not done and vector[2] != 0:
                        c = (-1-f[2])/vector[2]
                        if c > 0:
                            x = f[0]+c*vector[0]
                            y = f[1]+c*vector[1]
                            if (x >= -1 and x <= 1 and y >= -1 and y <= 1):
                                t = np.array([x, y, -1])
                                done = True
                    
                    #f
                    if not done and vector[2] != 0:
                        c = (1-f[2])/vector[2]
                        if c > 0:
                            x = f[0]+c*vector[0]
                            y = f[1]+c*vector[1]
                            if (x >= -1 and x <= 1 and y >= -1 and y <= 1):
                                t = np.array([x, y, 1])
                                done = True
                    
                    if done:
                        self.obj.all_edges.append([f, t])
    
    def norm(self):
        self.perspective_projection()
        self.draw_2D_graph()

    def image(self):
        self.draw_2D_graph(is_norm=False)
    
    def perspective_projection(self):
        #build perspective matrix
        matrix = np.matrix([[(2*self.cam_fru.n)/(self.cam_fru.r-self.cam_fru.l), 0, (self.cam_fru.r+self.cam_fru.l)/(self.cam_fru.r-self.cam_fru.l), 0],[0,(2*self.cam_fru.n)/(self.cam_fru.t-self.cam_fru.b),(self.cam_fru.t+self.cam_fru.b)/(self.cam_fru.t-self.cam_fru.b), 0],[0, 0, (-self.cam_fru.f-self.cam_fru.n)/(self.cam_fru.f-self.cam_fru.n),(-2*self.cam_fru.f*self.cam_fru.n)/(self.cam_fru.f-self.cam_fru.n)],[0,0,-1,0]])

        #perspective projection
        for idx, arr in enumerate(self.obj.all_edges):
            f, t = arr
            f = np.resize(f, 3)
            f = np.append(f, 1)
            t = np.resize(t, 3)
            t = np.append(t, 1)
            f = np.reshape(np.array(matrix.dot(np.matrix(f).T).T), 4)
            t = np.reshape(np.array(matrix.dot(np.matrix(t).T).T), 4)
            self.obj.all_edges[idx] = [f,t]

    def show_obj_points_in_UI(self):
        for i in range(5):
            command = 'self.ui.obj_{}.setText(self.np_array_to_string(self.obj.points[i]))'.format(i+1)
            exec(command)

    def draw_3d_graph(self, is_norm=False):
        plt.clf()
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        if is_norm:
            ax.set_xlim3d(-1, 1)
            ax.set_ylim3d(-1, 1)
            ax.set_zlim3d(-1, 1)
            #draw point
            for idx, point in enumerate(self.obj.points):
                if (self.obj.is_inside[idx]):
                    ax.scatter(point[0], point[1], point[2], c=self.obj.color[idx])
            #draw line
            for idx, point in enumerate(self.obj.all_edges):
                f, t = point
                ax.plot((f[0],t[0]), (f[1],t[1]), (f[2],t[2]), 'k')

        else :
            ax.set_xlim3d(-5, 5)
            ax.set_ylim3d(-5, 5)
            ax.set_zlim3d(-5, 5)
            ax.scatter(self.camera.x, self.camera.y, self.camera.z, c='r', label='camera')
            for idx, p in enumerate(self.obj.points):
                ax.scatter(p[0], p[1], p[2], c=self.obj.color[idx])
            for i in range(4):
                x1, y1, z1 = np.resize(self.obj.points[i], 3)
                for p in self.obj.lines[i]:
                    x2, y2, z2 = np.resize(self.obj.points[p], 3)
                    ax.plot((x1,x2), (y1, y2), (z1, z2), 'k')

        plt.savefig('g.png')
        img = QtGui.QPixmap('g.png')
        img = img.scaled(640, 600, QtCore.Qt.KeepAspectRatio)
        self.ui.draw_area.setPixmap(img)
        fig.clear()
        plt.close(fig)
    
    def draw_2D_graph(self, is_norm=True):
        plt.clf()
        plt.grid(True)

        if is_norm:
            x = y = 1
        else:
            x = (self.cam_fru.r - self.cam_fru.l)/2
            y = (self.cam_fru.t - self.cam_fru.b)/2

        for idx, point in enumerate(self.obj.all_edges):
            f, t = point
            plt.plot((x*f[0],x*t[0]), (y*f[1],y*t[1]), 'k')

        if is_norm:
            plt.xlim(-1, 1)
            plt.ylim(-1, 1)
        else:
            plt.xlim(self.cam_fru.l, self.cam_fru.r)
            plt.ylim(self.cam_fru.b, self.cam_fru.t)
        plt.savefig('g.png')
        img = QtGui.QPixmap('g.png')
        img = img.scaled(640, 600, QtCore.Qt.KeepAspectRatio)
        self.ui.draw_area.setPixmap(img)

    def np_array_to_string(self, obj):
        tmp = np.resize(obj, 3)
        s = ','.join(str(("{:.2f}".format(i))) for i in tmp)
        s = '(' + s + ')'
        return s

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = main_window()
    window.show()
    sys.exit(app.exec_())