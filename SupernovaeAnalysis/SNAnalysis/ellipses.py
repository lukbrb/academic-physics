from ast import Raise
import numpy as np
import matplotlib.pyplot as plt


class Ellipse:
    def __init__(self, chi2val=2.4477, data=None, covariance=None, average=None) -> None:
        self.chi2val = chi2val
        if data:
            self.data = data
            self.covariance = np.cov(data)
            self.eigenval, self.eigenvec = np.linalg.eig(covariance)
        elif covariance.all() is not None and average is not None:
            self.data = None
            self.covariance = covariance
            self.avg = average
            self.eigenval, self.eigenvec = np.linalg.eig(covariance)
        else:
            raise ValueError("Expected either data or covariance matrix and average but got none of those.")
    
    @property
    def eigenvalues(self):
        return self.eigenval
    
    @property
    def eigenvectors(self):
        return self.eigenvec


    def get_ellipse_properties(self):

        # Get the index of the largest eigenvector
        index_max = self.eigenval.argmax() # Probablement le vecteur associé à la plus grande valeur propre
        largest_eigenvec = self.eigenvec[index_max]
        largest_eigenval = self.eigenval.max()

        # Get the smallest eigenvalue and eigenvector 
        index_min = self.eigenval.argmin() # Probablement le vecteur associé à la plus grande valeur propre
        smallest_eigenvec = self.eigenvec[index_min]
        smallest_eigenval = self.eigenval.min()

        # print(eigenval, eigenvec)
        # print(largest_eigenvec, largest_eigenval)
        # print(smallest_eigenvec, smallest_eigenval)

        # Calculate the angle between the x-axis and the largest vector
        angle = np.arctan2(largest_eigenvec[1], largest_eigenvec[0])

        # The angle is between -pi and pi, we shift ot sit it's between 0 and 2pi
        if angle < 0:
            angle += 2 * np.pi 

        # Get the coordinates of the mean data
        if not self.average:
            self.avg = np.mean(self.data, axis=1)
        
        # Get the 95% confidence interval error ellipse
        theta_grid = np.linspace(0, 2 * np.pi)
        phi = -angle
        self.X0 = self.avg[0]
        self.Y0= self.avg[1]
        a = self.chisquare_val * np.sqrt(largest_eigenval)
        b = self.chisquare_val * np.sqrt(smallest_eigenval)

        # the ellipse in x and y coordinates 
        ellipse_x_r  = a * np.cos(theta_grid)
        ellipse_y_r  = b * np.sin(theta_grid)

        # Define a rotation matrix
        R = np.array([[np.cos(phi), np.sin(phi)], 
                    [-np.sin(phi), np.cos(phi)]
                    ])

        # let's rotate the ellipse to some angle phi
        self.r_ellipse = np.dot(np.array([ellipse_x_r, ellipse_y_r]).T,  R)

        return self.r_ellipse, self.X0, self.Y0

    def plot(self):
        # Draw the error ellipse
        plt.plot(self.r_ellipse[:,0] + self.X0, self.r_ellipse[:,1] + self.Y0,'-')
        if self.data.all() is not None:
            # Plot the original data
            plt.plot(self.data[0,:], self.data[1,:], '.')
        # plt.xlim([data.min() - 3, data.max() + 3])

        # Plot the eigenvectors
        plt.quiver(self.X0, self.Y0, self.largest_eigenvec[0] * np.sqrt(self.largest_eigenval), self.largest_eigenvec[1] * np.sqrt(self.largest_eigenval))
        plt.quiver(self.X0, self.Y0, self.smallest_eigenvec[0] * np.sqrt(self.smallest_eigenval), self.smallest_eigenvec[1] * np.sqrt(self.smallest_eigenval))
        plt.show()