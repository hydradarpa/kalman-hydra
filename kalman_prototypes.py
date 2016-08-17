
class KFStateMorph(KFState):
	def __init__(self, distmesh, im, flow, cuda, eps_Q = 1, eps_R = 1e-3):
		#Set up initial geometry parameters and covariance matrices
		self._ver = np.array(distmesh.p, np.float32)
		#Morph basis connecting mesh points to morph bases
		#T and K have the property that T*K = X (positions of nodes)
		self._generate_morph_basis(distmesh)

		self._vel = np.zeros(self.K.shape, np.float32)
		#For testing we'll give some initial velocity
		self._vel = np.ones(self.K.shape, np.float32)

		#Set up initial guess for texture
		self.tex = im
		self.nx = im.shape[0]
		self.ny = im.shape[1]
		self.M = self.nx*self.ny

		#Number of 'observations'
		self.NZ = self.M
		self.eps_Q = eps_Q
		self.eps_R = eps_R

		#Fixed quantities
		#Coordinates relative to texture. Stays constant throughout video
		self.N = self.K.shape[0]
		self.u = self._ver
		#Orientation of simplices
		self.tri = distmesh.t
		self.NT = self.tri.shape[0]
		#The SciPy documentation claims the edges are provided in an order that orients
		#them counter-clockwise, though this doesn't appear to be the case...
		#So we should keep the orientations here. Still not sure if the magnitude
		#of the cross product is useful or not. Probably not
		a = self._ver[self.tri[:,1],:] - self._ver[self.tri[:,0],:]
		b = self._ver[self.tri[:,2],:] - self._ver[self.tri[:,0],:]
		self.ori = np.sign(np.cross(a,b))

		#Form state vector
		self.X = np.vstack((self.K.reshape((-1,1)), self._vel.reshape((-1,1))))
		self.V = self.velocities().reshape((-1,1))
		e = np.eye(2*self.N)
		z = np.zeros((2*self.N,2*self.N))
		self.F = np.bmat([[e, e], [z, e]])
		self.Q = eps_Q * np.bmat([[e/4, e/2], [e/2, e]])
		self.R = eps_R * np.ones((self.NZ,self.NZ))
		self.P = np.eye(self._vel.shape[0]*4)

		#Renderer
		self.renderer = Renderer(distmesh, self.V, flow, self.nx, im, self.eps_R, cuda)

	def _generate_morph_basis(self, distmesh):
		#import rpdb2 
		#rpdb2.start_embedded_debugger("asdf")
		self.T = np.ones((distmesh.p.shape[0], 3))
		#K here is just translation of points
		self.K = np.mean(distmesh.p, axis = 0)
		self.T[:,0:2] = distmesh.p - self.K 

	def vertices(self):
		ver = self.X[0:2].reshape((-1,2))
		K = np.vstack((np.np.eye(2), ver))
		return np.dot(self.T,K)

	def velocities(self):
		vel = self.X[2:].reshape((-1,2))
		K = np.vstack((np.zeros((2,2)), vel))
		return np.dot(self.T,K)

class KalmanFilterMorph(KalmanFilter):
	def __init__(self, distmesh, im, flow, cuda):
		self.distmesh = distmesh
		self.N = distmesh.size()
		print 'Creating filter with ' + str(self.N) + ' nodes'
		self.state = KFStateMorph(distmesh, im, flow, cuda)

	def linearize_obs(self, z_tilde, y_im, deltaX = 2):
		H = np.zeros((self.state.M, self.size()))
		for idx in range(self.state.N*2):
			self.state.X[idx,0] += deltaX
			zp = self.observation(y_im)
			self.state.X[idx,0] -= deltaX
			H[:,idx] = (z_tilde - zp)/deltaX
		return H

class UnscentedKalmanFilter(KalmanFilter):
	def __init__(self, distmesh, im, flow, cuda):
		KalmanFilter.__init__(self, distmesh, im, flow, cuda)

		self.L=numel(x);                                 #numer of states
		self.m=numel(z);                                 #numer of measurements
		self.alpha=1e-3;                                 #default, tunable
		self.ki=0;                                       #default, tunable
		self.beta=2;                                     #default, tunable
		#scaling factor
		self.lmda=self.alpha*alpha*(self.L+self.ki)-self.L;
		self.c=self.L+self.lmda;                         #scaling factor

	def predict(self):
		#Setup sigma points and weights
		#Wm=[self.lmda/self.c, 0.5/self.c+zeros(1,2*self.L)];           #weights for means
		#Wc=Wm;
		#Wc(1)=Wc(1)+(1-self.alpha*self.alpha+self.beta);           #weights for covariance
		#c=sqrt(c);
		#X=sigmas(x,P,c);                            #sigma points around x
		#propagate sigma points 
		#[x1,X1,P1,X2]=ut(fstate,X,Wm,Wc,L,Q);		#unscented transformation of process
		return False 

	def update(self, y_im, y_flow = None):
		#[z1,Z1,P2,Z2]=ut(hmeas,X1,Wm,Wc,m,R)        #Unscented transformation of measurments
		#P12=X2*diag(Wc)*Z2.T                        #Transformed cross-covariance
		#K=P12*inv(P2)
		#x=x1+K*(z-z1)                               #State update
		#The problem here is that K is a NxM matrix, which is very large. 
		#Worse is that P2 is MxM... too big :(
		#P=P1-K*P12.T                                #Covariance update
		return False

	def ut(self,f,X,Wm,Wc,n,R):
		#Unscented Transformation
		#Input:
		#        f: nonlinear map
		#        X: sigma points
		#       Wm: weights for mean
		#       Wc: weights for covraiance
		#        n: numer of outputs of f
		#        R: additive covariance
		#Output:
		#        y: transformed mean
		#        Y: transformed smapling points
		#        P: transformed covariance
		#       Y1: transformed deviations
		L=size(X,2);
		y=zeros(n,1);
		Y=zeros(n,L);
		#for k in range(self.L):                   
		#	Y(:,k)=f(X(:,k));       
		#	y=y+Wm(k)*Y(:,k);       
		#Y1=Y-y(:,ones(1,L));
		#P=Y1*diag(Wc)*Y1.T+R;          
		return False #[y,Y,P,Y1]

	def sigmas(self,x,P,c):
		#Sigma points around reference point
		#Inputs:
		#       x: reference point
		#       P: covariance
		#       c: coefficient
		#Output:
		#       X: Sigma points		
		#A = c*chol(P).T
		#Y = x(:,ones(1,numel(x)))
		#X = [x Y+A Y-A]
		return False

#Mass-spring Kalman filter
class MSKalmanFilter(KalmanFilter):
	def __init__(self, distmesh, im, flow, cuda, sparse = True, multi = True, nI = 10, eps_F = 1, eps_Z = 1e-3, eps_J = 1e-3, eps_M = 1e-3):
		KalmanFilter.__init__(self, distmesh, im, flow, cuda, sparse = sparse, multi = multi, eps_F = eps_F, eps_Z = eps_Z, eps_J = eps_J, eps_M = eps_M)
		#Mass of vertices
		self.M = 1
		#Spring stiffness
		self.kappa = 1

	@timer(stats.statepredtime)
	def predict(self):
		print '-- predicting'
		X = self.state.X 
		self.orig_x = X.copy()
		F = self.state.F 
		Weps = self.state.Weps
		W = self.state.W 

		#Prediction equations 
		self.state.X = np.dot(F,X)
		self.pred_x = self.state.X.copy()
		self.state.W = np.dot(F, np.dot(W,F.T)) + Weps 
