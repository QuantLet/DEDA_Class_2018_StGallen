## Title:           Scagnostics of University Rankings
## Subtitle:        Analysis of Financial Times Rankings for the Master in Management
##
## Course:          Smart Data Analysis
## M.A. Students:   Lars Stauffenegger, Peter De Cour
## Lecturer:        Professor Wolfgang Haerdle (HU Berlin)
## Place, Time:     University of St. Gallen, 29.10.18 - 2.11.18 

#-------------------------------------------------------------

## Packages used

import pandas as pd
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.spatial import ConvexHull
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
import os

#-------------------------------------------------------------

## Web Scraping: ATTENTION FORMAT DIFFERENT IN EXCEL THAN MANUAL DOWNLOAD
## Manually downloaded File in same Folder: mmgmt18.csv
#url_wo_year='http://rankings.ft.com/businessschoolrankings/masters-in-management-'

#-------------------------------------------------------------

## Data Import and Cleansing

# Directories
#path = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..')) # in case the .py-file is in its own folder
path              = os.path.dirname(os.path.realpath("__file__")) + '\\' # .py-file in Parent Directory
mst_path          = path + '\\MST_Plots\\'
convex_hull_path  = path + '\\Convex_Hull_Plots\\'
concave_hull_path = path + '\\Concave_Hull_Plots\\'
ols_path          = path + '\\OLS_Summaries\\'
file              ='\\FT_MiM_Data\\mmgmt18.csv'

# Ranking column name
year        = "2018"
Ranking_Col = "FT_Rank " + year

# Import into Data Frame
data = pd.read_csv(path + file,encoding='ISO-8859-1', delimiter = ";")
df   = pd.DataFrame(data).rename(columns={year: Ranking_Col})

# Drop Columns with Text and NAN
df_num   = df.select_dtypes(['number'])
df_clean = df_num.dropna(axis=1)

# Shape of the cleaned data matrix
size   = df_clean.shape     # size of dataframe
nr_col = size[1]            # nr of columns

#-------------------------------------------------------------

## Definition of Functions

def Minimum_Spanning_Tree(X, copy_X=True):
    # Found Code on:
    # http://peekaboo-vision.blogspot.com/2012/02/simplistic-minimum-spanning-tree-in.html
    # Author: Andreas Mueller
    if copy_X:
        X = X.copy()
 
    if X.shape[0] != X.shape[1]:
        raise ValueError("X needs to be square matrix of edge weights")
    n_vertices     = X.shape[0]
    spanning_edges = []
     
    # initialize with node 0:                                                                                         
    visited_vertices = [0]                                                                                            
    num_visited = 1
    # exclude self connections:
    diag_indices = np.arange(n_vertices)
    X[diag_indices, diag_indices] = np.inf
     
    while num_visited != n_vertices:
        new_edge = np.argmin(X[visited_vertices], axis=None)
        # 2d encoding of new_edge from flat, get correct indices                                                      
        new_edge = divmod(new_edge, n_vertices)
        new_edge = [visited_vertices[new_edge[0]], new_edge[1]]                                                      
        # add edge to tree
        spanning_edges.append(new_edge)
        visited_vertices.append(new_edge[1])
        # remove all edges inside current tree
        X[visited_vertices, new_edge[1]] = np.inf
        X[new_edge[1], visited_vertices] = np.inf                                                                     
        num_visited += 1
    return np.vstack(spanning_edges)

   
def MST_Plot(P,plot_path,indep_var):
    # Same author as function above; Modified for this Project by Lars 
    # MST Edges
    X = squareform(pdist(P))
    edge_list = Minimum_Spanning_Tree(X)
    
    # MST scatter plot
    plt.scatter(P[:, 0], P[:, 1])
    for edge in edge_list:
        i, j = edge
        plt.plot([P[i, 0], P[j, 0]], [P[i, 1], P[j, 1]], c='r')
    
    plt.xlabel(indep_var, fontsize=18)
    ax = plt.gca()
    ax.set_aspect('equal')
    ax.set_xlim([-5, 105])
    ax.set_ylim([105, -5])
    
    # filepath for plot image
    figure_path = plot_path + 'MST_' + indep_var.replace("/", "-") + '.png'
    plt.savefig(figure_path, bbox_inches='tight')
    
    # Use close or show command!
    #plt.close()
    plt.show()

  
def Rescale_Indep_Var(M,indep_var):
    # Rescale Independent Variable between 0 and 100 if not rank or percentage
    if 'rank' in indep_var:
        print(indep_var, "is on a scale from 1 to 100.")
    elif '%' in indep_var:
        print(indep_var, "is on a scale from 0 to 100.")
    else:
        X_min = np.amin(M[:,0])
        X_max = np.amax(M[:,0])
        M[:,0] = (M[:,0] - X_min) / (X_max - X_min) * 100
        indep_var = 'normalised ' + indep_var
        print(indep_var)
    
    return M
    

def Convex_Hull(M,convex_hull_path,indep_var):
    # Convex Hull
    hull = ConvexHull(M)
    
    # Coordinates of the Convex Hull
    cx = np.array(hull.points[hull.vertices,0])
    cy = np.array(hull.points[hull.vertices,1])
    Cord = np.column_stack([cy,cx])
    
    # Euklidean Distances Matrix   
    D = Eukl_Dist(Cord)
    
    Convex_Hull_sum = 0
    # Sum of all edges of the Convex Hull
    for i in range(0,D.shape[1]-1):
        Convex_Hull_sum = Convex_Hull_sum + D[i,i+1]
    
    # Area/Surface of the Alpha Space
    Convex_Hull_area = Polygon_Area(Cord)
    
    # Convex Hull Plots
    plt.plot(M[:,0], M[:,1], 'o')
    plt.xlabel(indep_var, fontsize=18)
    axh = plt.gca()
    axh.set_aspect('equal')
    axh.set_xlim([-5, 105])
    axh.set_ylim([105, -5])
    
    for simplex in hull.simplices:
        plt.plot(M[simplex, 0], M[simplex, 1], 'k-')
    
    figure_path = convex_hull_path + 'Convex_Hull_' + indep_var.replace("/", "-") + '.png'
    plt.savefig(figure_path, bbox_inches='tight')
    
    # Must use close or show command! (to delete "background" variable)
    #plt.close()
    plt.show()
    
    # Output Variables
    return Convex_Hull_sum, Convex_Hull_area

def Concave_Hull(M,alpha_x,alpha_y,concave_hull_path,indep_var):
    # Delauney Triangulation and Plots
    de = Delaunay(M)
    plt.triplot(M[:,0], M[:,1], de.simplices.copy())
    plt.plot(M[:,0], M[:,1], 'o')
    plt.xlabel(indep_var, fontsize=18)
    axh = plt.gca()
    axh.set_aspect('equal')
    axh.set_xlim([-5, 105])
    axh.set_ylim([105, -5])
    plt.show() # or plt.close()
    
    # Initializing length < alpha
    dec = []
    a = alpha_x
    b = alpha_y
    
    # Simplex of all triangles with edge length < alpha & Plots
    for i in de.simplices:
        tmp = []
        j = [M[c] for c in i]
        if abs(j[0][1] - j[1][1])>a or abs(j[1][1]-j[2][1])>a or abs(j[0][1]-j[2][1])>a or abs(j[0][0]-j[1][0])>b or abs(j[1][0]-j[2][0])>b or abs(j[0][0]-j[2][0])>b:
            continue
        for c in i:
            tmp.append(M[c])
        dec.append(tmp)

    # Concave Hull PLots
    plt.plot(M[:,0], M[:,1], 'o')
    plt.xlabel(indep_var, fontsize=18)
    axh = plt.gca()
    axh.set_aspect('equal')
    axh.set_xlim([-5, 105])
    axh.set_ylim([105, -5])
    for i in range(0,np.shape(dec)[0]):
        plt.plot([dec[i][0][0], dec[i][1][0]], [dec[i][0][1], dec[i][1][1]], c='r')
        plt.plot([dec[i][0][0], dec[i][2][0]], [dec[i][0][1], dec[i][2][1]], c='r')
        plt.plot([dec[i][1][0], dec[i][2][0]], [dec[i][1][1], dec[i][2][1]], c='r')
    figure_path = concave_hull_path + 'Concave_Hull_' + indep_var.replace("/", "-") + '.png'
    plt.savefig(figure_path, bbox_inches='tight')
    
    # Must use close or show command! (to delete "background" variable)
    #plt.close()
    plt.show()
    
    
def Eukl_Dist(M):
    # Euclidean Distances: Alternative to squareform(pdist(P)) 
    # Numpy Matrix with dependent and independent variable
    size = M.shape         # size of the array
    n    = size[0]         # number of observations in M / dots in plot
    D    = np.zeros((n,n)) # assign space for Euclidean Distances Matrix
    
    ## Create Euclidean Distances Matrix
    for p in range(0,n):
        x1 = M[p,1]
        y1 = M[p,0]
        for q in range(0,n):
            x2 = M[q,1]
            y2 = M[q,0]
            D[p,q] = np.sqrt(np.square(x1 - x2) + np.square(y1 - y2))
    
    return D


def Polygon_Area(corners):
    # Shoelace formula for are of Polygon
    # http://code.activestate.com/recipes/578047-area-of-polygon-using-shoelace-formula/
    n = len(corners) # of corners
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += corners[i][0] * corners[j][1]
        area -= corners[j][0] * corners[i][1]
    area = abs(area) / 2.0
    return area


#-------------------------------------------------------------

## Minimum Spanning Tree, Convex Hull and Concave Hull

# Initializing Variables  
Convex_Hull_Edges_Sum = np.zeros((nr_col)) 
Convex_Hull_Area = np.zeros((nr_col)) 

## MST, Convex and Concave Hull looping through our Data Sets
for c in range (1,nr_col):
    # Col Name
    indep_var = df_clean.columns[c]
      
    # Prepare Dependendt and Independent Variable in 2 Column Vector
    M = np.array(df_clean[[indep_var,Ranking_Col]])
    
    #Rescale Independent Variable if not a Rank or a Percentage
    Rescale_Indep_Var(M,indep_var)

    # Run Plots
    MST_Plot(M,mst_path,indep_var)

    # Convex Hull includes saving the Plot
    # Edge Sum and Total Surface
    # First Entry will be zero as Dependent Variable with itself has no point cloud
    Convex_Hull_Edges_Sum[c], Convex_Hull_Area[c] = Convex_Hull(M,convex_hull_path,indep_var)
    
    # ALpha Shape resp Concave Hull
    # Alpha is the Maximum distance between two pointsare allowed to have on the x and on the y axis.
    alpha_x = 40
    alpha_y = 40
    Concave_Hull(M,alpha_x,alpha_y,concave_hull_path,indep_var)
    # TO DO: How to determine alphas?
    # TO DO: Get rid off edges within Alpha Shape.
    # TO DO: Determine Edges_Sum and Area.
       
#------------------------------------------------------------- 

## TO DO: Scagnostic Key Figure C for different Shapes (so far only R Package in Java)

#-------------------------------------------------------------
    
### Linear Regression: Ordinary Least Square
#c = #COLNUMBER
## Col Name
#indep_var = df_clean.columns[c]
#
## Variables 
#X = df_clean[indep_var]
#X = sm.add_constant(X)
#Y = df_clean[Ranking_Col]
#
## OLS regression
#model = sm.OLS(Y,X)
#rm = model.fit()
#
## Output
#print (rm.params)
#print (rm.summary())
#
## Save Output
#plt.rc('figure', figsize=(12, 7))
#plt.text(0.01, 0.05, str(rm.summary()), {'fontsize': 10}, fontproperties = 'monospace') # approach improved by OP -> monospace!
#plt.axis('off')
#plt.tight_layout()
#plt.savefig(ols_path + 'OLS_' + indep_var.replace("/", "-") + '.png', bbox_inches='tight')

#-------------------------------------------------------------

## TO BE FINALIZED (exceeded the time constraint of this project)
#
## To Do: First Steps for Scagnostics Measures
##find_outliers
#def find_cutoff(MST):
#    q25                     = np.quantile(MST,0.25)
#    q75                     = np.quantile(MST,0.75)
#    cutoff                  = q75 - 1.5 * (q75 - q25)
#    
#    return cutoff
#    
#def minimum_edge(tuple_array):
#    min_edge = []
#    for o in range(len(tuple_array)):
#        edges_length = []
#        m,n = tuple_array[o]
#            
#        for l in range(len(tuple_array)):
#            i,j = tuple_array[l]
#            edges_length.append(np.sqrt((m-i)**2+(n-j)**2))
#        edges_length.pop(o)
#        min_edge.append(min(edges_length))
#    return min_edge
#    
## outliers dictionary
#def compute_outliers_dict():
#    outliers_dict = {}
#    for k in df_clean.columns:
#        outliers_dict[k] = {}
#        
#        tuple_array = []
#        for i in range(len(df["2018"])):
#            tuple_array.append((df["2018"][i],df[k][i]))    
#        points = np.asarray(df[[k,"2018"]])
#        
#        me = minimum_edge(tuple_array)
#        total_edge_length = sum(me)
#        
#        school_names =  np.asarray(df["School name"])
#        me_school   = pd.DataFrame({'ME':me, 'School name':school_names})
#        me_school.sort_values(by="ME")
#        
#        cut = find_cutoff(me_school["ME"])
#        outliers            = []
#        outliers_schools    = []
#        MST_outliers_length = 0
#    
#        # find outliers
#        for i in range(len(points)):
#            if me[i] < cut:
#                outliers.append(i)
#                outliers_schools.append(me_school["School name"][i])
#                MST_outliers_length += me[i]
#        # c for outlier
#        outlier_c = MST_outliers_length /  total_edge_length 
#        
#        outliers_dict[k]["Schools"]          = outliers_schools
#        outliers_dict[k]["Outliers"]         = outliers
#        outliers_dict[k]["Outlier Measure"]  = outlier_c
#        outliers_dict[k]["ME"]               = me
#        outliers_dict[k]["Total_Length"]     = total_edge_length
#    return outliers_dict
## Calculate Alpha Value
#def alpha_value():
#    q90     = np.quantile(me_school["ME"],0.90)
#    alpha   = q90
#    return min(alpha,100)
#        
## Sparsity
#def calculate_sparsity(minimum_edge):
#    q90      = np.quantile(minimum_edge,0.9)
#    sparsity = min(q90 / 1000, 1)
#    return sparsity
## skewness: ratio of edge lengths in edge distribution: (q90 - q50) / (q90 - q10)
#def calculate_skewness(minimum_edge):
#    q90                 = np.quantile(minimum_edge,0.9)
#    q50                 = np.quantile(minimum_edge,0.5)
#    q10                 = np.quantile(minimum_edge,0.1)
#    skewness            = (q90 - q50) / (q90 - q10)
#    # !!!!!!!
#    #t                   = len(minimum_edge) / 500 # total count of binned data
#    # !!!!!!!
#    #correction          = 0.7 + 0.3 * (1 + t**2)
#    
#    #corrected_skewness  = 1 - correction * (1 - skewness) 
#    #return corrected_skewness
#    return skewness
#    
## clumpy = max(1-max(legnth(ek))/length(ej)) - 
#    
#    
## skinny = 1 - sqrt(4pi * area(a))) / perimeter(a)  
#    
# 
## convex = area(A)/area(H) - area of alpha hull / convex hull