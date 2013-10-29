# coding:utf-8
import numpy as np
import decimal
import os.path
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter


def qform(x,y,min_norm):
	return np.sum([np.prod([min_norm,np.power(x,2)]),
		np.prod([min_norm,np.power(y,2)]),
		np.prod([2,x,y,min_norm,np.sqrt(1.0-1.0/np.power(min_norm,2))])])


def lcm(a,b):
	greatest_common_divisor = gcd(a,b)
	return int(np.prod([a/greatest_common_divisor,b]))

def gcd(a,b):
	if a==0 or b==0:
		return 1
	elif a==b and a>0 and b>0:
		return a
	if a%b == 0 and a>0 and b>0:
		return b
	elif b%a == 0 and a>0 and b>0:
		return a
	elif np.prod([100000,a])<b and a>0 and b>0:
		return gcd(a,np.sum([b,-np.prod([1000,a])]))
	elif np.prod([10000,a])<b and a>0 and b>0:
		return gcd(a,np.sum([b,-np.prod([1000,a])]))
	elif np.prod([1000,a])<b and a>0 and b>0:
		return gcd(a,np.sum([b,-np.prod([1000,a])]))
	elif np.prod([100,a])<b and a>0 and b>0:
		return gcd(a,np.sum([b,-np.prod([100,a])]))
	elif np.prod([10,b])<b and a>0 and b>0:
		return gcd(a,np.sum([b,-np.prod([10,a])]))
	elif a<b and a>0 and b>0:
		return gcd(a,np.sum([b,-a]))
	elif a>np.prod([100000,b]) and a>0 and b>0:
		return gcd(np.sum([a,-np.prod([100000,b])]),b)
	elif a>np.prod([10000,b]) and a>0 and b>0:
		return gcd(np.sum([a,-np.prod([10000,b])]),b)
	elif a>np.prod([1000,b]) and a>0 and b>0:
		return gcd(np.sum([a,-np.prod([1000,b])]),b)
	elif a>np.prod([100,b]) and a>0 and b>0:
		return gcd(np.sum([a,-np.prod([100,b])]),b)
	elif a>np.prod([10,b]) and a>0 and b>0:
		return gcd(np.sum([a,-np.prod([10,b])]),b)
	elif a>b and a>0 and b>0:
		return gcd(np.sum([a,-b]),b)
	elif a<0 and b<0:
		return gcd(-a,-b)
	elif a<0:
		return gcd(-a,b)
	elif b<0:
		return gcd(a,-b)

def decimal_to_fraction(dec):
	# returns irreducible fraction of the form a/b
	if dec>1:
		return [0,0]
	elif dec == 1:
		return [1,1]
	else:
		digits_after_decimal_point = -1*decimal.Decimal(str(dec)).as_tuple().exponent
		digits_after_decimal_point = digits_after_decimal_point if digits_after_decimal_point<7 else 6


		b = np.power(10,digits_after_decimal_point)
		a = int(np.prod([dec,b]))
		greatest_common_divisor = gcd(a,b)
		b /= greatest_common_divisor
		a /= greatest_common_divisor
		return [int(a),int(b)]

def lattice_zeta_function(s,lattice,limits):
	return zeta_function(s,lattice[0,0],limits)

def zeta_function (s,min_norm,limits):
	total = 0.0
	for x in range(-1*limits,limits):
		for y in range(-1*limits,limits):
			if x!=0 or y!=0:
				p = np.power(qform(x,y,min_norm),np.prod([s,2]))
				total+= 1.0/(p)
	return total


def plot_zeta_function(s, n_steps):
	xlabel = '|W|'
	ylabel = 'E(W,%s)' % s
	color  = 'k.'
	X = np.arange(1.0, 1.1547005383792515, 0.1547005383792515/float(n_steps))
	Y = []
	for norm_index in range(0,len(X)):
		lattice = fixed_determinant_wr_lattice(X[norm_index])
		zeta = lattice_zeta_function(s,lattice,15)
		Y.append(zeta)
	fig = plt.figure(figsize=(8,4), dpi=100)
	axes = fig.add_axes([0.1, 0.1, 0.8, 0.8]) # lower, bottom, width, height (range 0 to 1)
	axes.plot(X, Y, '#272822')
	if n_steps<100:
		axes.plot(X, Y, 'k.')
	axes.set_xlabel(xlabel)
	axes.set_ylabel(ylabel)
	fig.savefig("./../thesis_images/zeta_%s_.svg" % s)
	plt.show()

def random_zeta_function(s, k):
	xlabel = '|W|'
	ylabel = 'E(W,%s)' % s
	color  = 'k.'

	X = []
	for i in range(0,k):
		# np.random.random_sample() is a numpy method
		# for a uniform random variable U(0,1)
		X.append(1.0+0.1547005383792515*np.random.random_sample())
	X.sort()
	Y = []
	for norm_index in range(0,len(X)):
		lattice = fixed_determinant_wr_lattice(X[norm_index])
		zeta = lattice_zeta_function(s,lattice,15)
		Y.append(zeta)
	fig = plt.figure(figsize=(8,4), dpi=100)
	axes = fig.add_axes([0.1, 0.1, 0.8, 0.8]) # lower, bottom, width, height (range 0 to 1)
	axes.plot(X, Y, '#272822')
	axes.plot(X, Y, 'k.')
	# if k<100:
		# axes.plot(X, Y, 'k.')
	axes.set_xlabel(xlabel)
	axes.set_ylabel(ylabel)
	iteration = 0
	while True:
		if os.path.exists("./../thesis_images/random_sample_size_k_%s_%s.svg" %(s, iteration)):
			iteration +=1
		else:
			break
	fig.savefig("./../thesis_images/random_sample_size_k_%s_%s.svg" %(s, iteration))
	plt.show()
	

def plot_zeta_functions(s_steps,n_steps):
	fig = plt.figure()
	ax = fig.gca(projection='3d')
	X = np.arange(2.0, 3.0, 1.0/float(s_steps))
	Y = np.arange(1.0, 1.1547005383792515, 0.1547005383792515/float(n_steps))
	X, Y = np.meshgrid(X, Y)
	Z = []
	max_zeta_value = 0.0
	for s_index in range(0,len(X)):
		row = []
		for norm_index in range(0,len(Y)):
			lattice = fixed_determinant_wr_lattice(Y[s_index][norm_index])
			zeta = lattice_zeta_function(X[s_index][norm_index],lattice,50)
			max_zeta_value = max_zeta_value if zeta <= max_zeta_value else zeta
			row.append(zeta)
		Z.append(row)
	Z = np.array(Z)
	surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.YlGnBu,
	        linewidth=0, antialiased=True)
	ax.set_zlim(0, max_zeta_value*1.1)

	ax.zaxis.set_major_locator(LinearLocator(10))
	ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

	fig.colorbar(surf, shrink=0.5, aspect=5)
	fig.suptitle('WR Lattice Epstein zeta-function (W,s), s in [2-3], det(W)=1, |W| in [1-1.1547]', fontsize=20)
	plt.xlabel('s', fontsize=18)
	plt.ylabel('|W|', fontsize=16)

	plt.show()

def get_minimal_wr_lattice (angle):
	cos = np.cos(angle)
	sin = np.sin(angle)
	p_over_q_fraction = decimal_to_fraction(cos)
	r_rootD_over_q_fraction = decimal_to_fraction(sin)
	
	least_common_multiple      = lcm(p_over_q_fraction[1],r_rootD_over_q_fraction[1])
	p_over_q_fraction[0]       = np.prod([p_over_q_fraction[0],least_common_multiple/p_over_q_fraction[1]])
	p_over_q_fraction[1]       = least_common_multiple
	
	r_rootD_over_q_fraction[0] = np.prod([r_rootD_over_q_fraction[0],least_common_multiple/r_rootD_over_q_fraction[1]])
	r_rootD_over_q_fraction[1] = least_common_multiple


	# to debug values uncomment these lines:

	# print "     p          "+str(p_over_q_fraction[0])
	# print " ---------  =   ----------"
	# print "     q          "+str(p_over_q_fraction[1])
	# print ""
	# print " r(D)^0.5       "+str(r_rootD_over_q_fraction[0])
	# print " ---------  =   ----------"
	# print "     q          "+str(r_rootD_over_q_fraction[1])

	one_over_root_q = 1/np.sqrt(p_over_q_fraction[1])
	minimal_matrix = np.matrix([
		[np.prod([p_over_q_fraction[1],one_over_root_q]),
		np.prod([p_over_q_fraction[0],one_over_root_q])],
		[0,
		np.prod([r_rootD_over_q_fraction[0],one_over_root_q])]
		])

	# print minimal_matrix
	return minimal_matrix

# def increase_minimal_norm(matrix):
# 	determinant = np.linalg.det(matrix) # r D^0.5

	# because determinant of product of matrices is product of determinants
	# we keep determinant same by multiplying by a matrix U with det(U)=1

def wr(angle):
	return get_minimal_wr_lattice(angle)

def compare_lattices(s,angle_range,steps):
	increments = (angle_range[1]-angle_range[0])/float(steps)
	zeta_values = {}
	conversion_to_deg = 180.0/np.pi
	with open('lattice_angles.txt', 'w+') as f:
		f.write('				Minimal Integral Well Rounded Lattice Epstein zeta-function values for s=%s' % s)
		f.write('\n\n')
		for i in range(0,steps):
			angle_to_test = np.sum([angle_range[0],np.prod([i,increments])])
			min_lattice = wr(angle_to_test)
			zeta_values[angle_to_test] = zeta_function(s,min_lattice[0,0],15)
			f.write('%s°	E_Λ(%s) = %s\n' %("%.2f" % np.prod([angle_to_test,conversion_to_deg]), s, zeta_values[angle_to_test]))

def fixed_determinant_wr_lattice(norm):
	# determinant = 1.0 in a wr lattice in preferred orientation:
	# 	=> x only has 1 dimension
	#   => values between 1 and 1.1547005384 makes sense for WR (pi/3 to pi/2)
	if norm>1.1548:
		return fixed_determinant_wr_lattice(1.1547005383792515)
	if norm<1.0:
		return fixed_determinant_wr_lattice(1.0)
	sin_angle = (1.0/np.power(norm,2))
	cos_angle = np.power(np.sum([1.0,-np.power(sin_angle,2)]),0.5)
	wr_lattice = np.matrix([
		[norm, np.prod([cos_angle,norm])],
		[0,    np.prod([sin_angle,norm])]
		])
	return wr_lattice

def compare_det_lattices(s,steps):
	step_size = 0.1547005383792515/(steps+1.0) #1.1547005383792515290182975610039149112952035025402537
	minimal_norm = 1.0
	with open('lattice_det_1_s_'+str(s)+'.txt', 'w+') as f:
		f.write('				Well Rounded Lattice Epstein zeta-function values for s=%s, and det(Λ)=1' % s)
		f.write('\n\n')
		for i in range(0,steps+1):
			lattice = fixed_determinant_wr_lattice(minimal_norm)
			zeta = lattice_zeta_function(s,lattice,15)
			f.write('|Λ| = %s	E_Λ(%s) = %s\n' %("%.2f" % minimal_norm, s, zeta))
			minimal_norm += step_size

