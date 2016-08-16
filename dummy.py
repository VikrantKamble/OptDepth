"""
globInd = np.where((wl > 1500) & (wl < 1600))[0]
wave = wl[globInd]
for i in range(sum(w)):
	clf()
	spec = f[w][i]
	#myspec = A[0][i,:][0][globInd] - myfunct(wave, amp[i], alpha[i])
	#ivar = np.sqrt(A[1][i,:][0][globInd])
	plot(wl, spec, linewidth=0.4, color='k')
	#plot(wave, spec+3, linewidth=0.4, color='k')
	#xlim(1500,1600)
	temp = param[w][i]
	model = g2(wave,temp[0], temp[1],temp[2],temp[3],temp[4],temp[5])
	cont = myfunct(wave, amp[w][i], alpha[w][i])
	plot(wave, model+cont, linewidth=2.0, color='red')
	plot(wl, myfunct(wl, amp[w][i], alpha[w][i]),linewidth=2.0, color='red')
	#axhline(0)
	raw_input()


for i in t:
	clf()
	print "i=%d, chisq=%f" %(i, chisq[i]/dof[i])
	spec, ivar = A[0][i,:][0], A[1][i,:][0]
	#plot(wl, spec, linewidth=0.4, color='blue', alpha=0.4)
	#plot(wl, amp_o[i]*(wl/1450)**alpha_o[i], color='red')
	theta = list(param[i]) + [amp_n[i], alpha_n[i]]
	plot(wl, model(wl, *theta), color='black')
	a = estimate(wl, spec, ivar, wav1, wav2, 3)
	estimate(wl, spec, ivar, wav1, wav2, 5)
	to = raw_input()
"""

W, S, I = np.zeros((3,0))

for j in range(len(wav1)):
	temp = np.where((wl > wav1[j][0]) & (wl < wav1[j][1]))[0]
	tempS, tempI  = spec[temp], ivar[temp]
	
	#Mask out narrow absorption lines
	cut = np.percentile(tempS, per_value)
	blah = np.where((tempS > cut) & (tempI > 0))[0]
	wave = wl[temp][blah]

	W = np.concatenate((W, wave))
	S = np.concatenate((S, tempS[blah]))
	I = np.concatenate((I, tempI[blah]))

temp = np.where((wl > wav2[0]) & (wl < wav2[1]))[0]
mywl, myspec, myivar = wl[temp], spec[temp], ivar[temp]

s = np.where(myivar > 0)[0]

# Smoothing to remove narrow absoprtion troughs, more than 3 sigma away 
smooth = convolve(myspec[s], Box1DKernel(20))
t = s[np.where(np.sqrt(myivar[s])*(smooth - myspec[s]) < 3)[0]]

W = np.concatenate((W, mywl[t]))
S = np.concatenate((S, myspec[t]))
I = np.concatenate((I, myivar[t]))

err = lambda p: np.mean(I*(model(W,*p) - S)**2)


A1, A2 = max(smooth), max(smooth)/3.0
C1 = C2 = mywl[s[np.argmax(smooth)]]
S1 = np.sqrt(np.sum((mywl[s]-C1)**2 * smooth)/np.sum(smooth))
S2 = S1*2.0

p_init = [A1,A2,S1,S2,C1,C2,1,-1]
bounds=[(0,None),(0,None),(0,50),(0,80),(1500,1600),(1500,1600),(0,None),(None,None)]

p_opt = minimize(
	err,
	p_init,
	bounds = bounds,
	method="L-BFGS-B"
	).x


for i in range(5):
    spec = COMP[i]
    ivar = IVAR[i]
    compute_alpha(wl, spec, ivar, wav_range, per_value)
    to = raw_input()
for i in t:
	clf()
	plot(wl, spec[i], linewidth=0.4, color='black')
	plot(wl, myfunct(wl, amp_o[i], alpha_o[i]), color='green')
	plot(wl, myfunct(wl, amp_n[i], alpha_n[i]), color='red')
	to = raw_input()

figure = corner.corner(comb, labels=[r"$CIV_o$", r"$CIV_n$"],
                         truths=[0.0, 0.0],
                         quantiles=[0.16, 0.5, 0.84],
                         show_titles=True, title_kwargs={"fontsize": 12})
figure.gca().annotate("Old CIV vs new", xy=(0.5, 1.0), xycoords="figure fraction",
                      xytext=(0, -5), textcoords="offset points",
                      ha="center", va="top")
figure.savefig("demo.png")


levels = np.linspace(1000,6500,10)
t = where((alpha > -6) & (alpha < 4) & (ew > 0) & (ew < 200))[0]
H, x_edges, y_edges = np.histogram2d(alpha[t], ew[t], bins=[np.linspace(-6,4,20),np.linspace(0,200,20)])
extent = [x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]]
scatter(alpha[t], ew[t], marker='+', alpha=0.4, color='gray')
contour(H.transpose(), extent=extent, linewidth=50,  linestyles='solid', levels=levels)
colorbar()

a_edges = np.linspace(-3.5, -0.4, 4)
c_edges = np.zeros((3,4))
for i in range(3):
	t = where((alpha > a_edges[i]) & (alpha < a_edges[i+1]))[0]
	c = histogram(ew[t], bins=np.linspace(0,200,20))
	temp = where(c[0]>1000)[0]
	c_edges[i] = np.linspace(c[1][min(temp)], c[1][max(temp)], 4)

from matplotlib import rc
rc('text', usetex=True)
rc('xtick', labelsize=20) 
rc('ytick', labelsize=20)
plottau('opt_V3_11', 2.2, 0,  'black')
plottau('opt_V3_12', 2.2, 0.2,  'black')
plottau('opt_V3_13', 2.2, 0.4,  'black')
plottau('opt_V3_21', 2.2, 0.6,  'black')
plottau('opt_V3_22', 2.2, 0.8,  'black')
plottau('opt_V3_23', 2.2, 1,  'black')
plottau('opt_V3_31', 2.2, 1.2,  'black')
plottau('opt_V3_32', 2.2, 1.4,  'black')
plottau('opt_V3_33', 2.2, 1.6,  'black')
minorticks_on()

