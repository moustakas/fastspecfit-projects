def _niceparnames(parnames):
    """Replace parameter names with nice names."""

    old = list(['tau',
           'tage',
           'mass',
           'logmass',
           'logzsol',
           'dust2'])
    new = list([r'$\tau$ (Gyr)',
           'Age (Gyr)',
           r'$M / M_{\odot}$',
           r'$\log_{10}\,(M / M_{\odot})$',
           r'$\log_{10}\, (Z / Z_{\odot})$',
           r'$\tau_{diffuse}$'])

    niceparnames = list(parnames).copy()
    for oo, nn in zip( old, new ):
        this = np.where(np.in1d(parnames, oo))[0]
        if len(this) > 0:
            niceparnames[this[0]] = nn
            
    return np.array(niceparnames)

def subtriangle(results, showpars=None, truths=None, start=0, thin=2,
                chains=slice(None), logify=None, extents=None, png=None,
                **kwargs):
    """Make a triangle plot of the (thinned, latter) samples of the posterior
    parameter space.  Optionally make the plot only for a supplied subset of
    the parameters.

    :param start:
        The iteration number to start with when drawing samples to plot.

    :param thin:
        The thinning of each chain to perform when drawing samples to plot.

    :param showpars:
        List of string names of parameters to include in the corner plot.

    :param truths:
        List of truth values for the chosen parameters
    
    """
    import corner as triangle

    # Get the ull out the parameter names and flatten the thinned chains.
    parnames = np.array(results['theta_labels'])
    print(parnames)

    # Restrict to a particular set of parameters.
    if showpars:
        ind_show = np.array([parnames.tolist().index(p) for p in showpars])
        parnames = parnames[ind_show]
    else:
        ind_show = slice(None)

    # Get the arrays we need (trace, wghts)
    trace = results['chain'][..., ind_show]
    if trace.ndim == 2:
        trace = trace[None, :]
    trace = trace[chains, start::thin, :]
    wghts = results.get('weights', None)
    if wghts is not None:
        wghts = wghts[start::thin]
    samples = trace.reshape(trace.shape[0] * trace.shape[1], trace.shape[2])

    # logify some parameters
    xx = samples.copy()
    if truths is not None:
        xx_truth = np.array(truths).copy()
    else:
        xx_truth = None
    if logify:
        for p in logify:
            if p in parnames:
                idx = parnames.tolist().index(p)
                xx[:, idx] = np.log10(xx[:,idx])
                parnames[idx] = "log({})".format(parnames[idx])
                if truths is not None:
                    xx_truth[idx] = np.log10(xx_truth[idx])

    # Make nice labels.
    niceparnames = _niceparnames(parnames)
        
    # mess with corner defaults
    corner_kwargs = {"plot_datapoints": False, "plot_density": False,
                     "fill_contours": True, "show_titles": True}
    corner_kwargs.update(kwargs)
    
    fig = triangle.corner(xx, labels=niceparnames, truths=xx_truth,
                          quantiles=[0.25, 0.5, 0.75], weights=wghts,
                          color='k', **corner_kwargs)

    if png:
        print('Writing {}'.format(png))
        fig.savefig(png)

        ###################################################
        ## P(M, SFR)
        #print('Hack!')
        #png = os.path.join(ngc5322dir, '{}-{}-pofm.png'.format(args.prefix, args.priors))
        #chain = result['chain']
        #lnprobability = result['lnprobability']
        #
        ## infer the SFR
        #sfr = np.zeros_like(lnprobability)
        #for ii in np.arange(len(lnprobability)):
        #    _, _, _ = model.mean_model(chain[ii, :], obs=obs, sps=sps)
        #    sfr[ii] = sps.ssp.sfr * model.params['mass']
        #
        #pdb.set_trace()
        #
        ##ax.set_xlabel(r'$\log_{10}({\rm Stellar\ Mass})\ (\mathcal{M}_{\odot})$')
        ##ax.set_ylabel(r'Marginalized Posterior Probability')
        #
        #fig, ax = plt.subplots(figsize=(8, 6))
        #ax.hist(chain[:, 4], bins=50, histtype='step', linewidth=2, edgecolor='k',fill=True)    
        #ax.set_xlim(11, 11.8)
        #ax.set_yticklabels([])
        #ax.set_xlabel(r'$\log_{10}(\mathcal{M}/\mathcal{M}_{\odot})$')
        #ax.set_ylabel(r'$P(\mathcal{M})$')
        #ax.xaxis.set_major_locator(MultipleLocator(0.2))
        ##for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
        ##         ax.get_xticklabels() + ax.get_yticklabels()):
        ##    item.set_fontsize(22)
        #print('Writing {}'.format(png))
        #plt.subplots_adjust(left=0.1, right=0.95, bottom=0.18, top=0.95)
        #fig.savefig(png)

        # Corner plot.
        png = os.path.join(ngc5322dir, '{}-{}-corner.png'.format(args.prefix, args.priors))
        subtriangle(result, showpars=['logmass', 'sfr', 'tau', 'dust2', 'dust_ratio'],
                    logify=['tau'], png=png)

        pdb.set_trace()

        #reader.subcorner(result, start=0, thin=1, fig=plt.subplots(5,5,figsize=(27,27))[0])

def bestfit_sed(obs, chain=None, lnprobability=None, theta=None, sps=None,
                model=None, seed=None, nrand=100, png=None):
    """Plot the (photometric) best-fitting SED.

    Either pass chain and lnprobability (to visualize the emcee fitting results)
    *or* theta (to visualize just a single SED fit).

    """
    # Get the galaxy photometry and filter info.
    weff, fwhm, galphot, galphoterr, mask = _galaxyphot(obs)

    # Build the maximum likelihood model fit and also grab a random sampling of
    # the chains with weight equal to the posterior probability.    
    if chain is not None:
        if chain.ndim == 3: # emcee
            nwalkers, niter, nparams = chain.shape
            ntot = nwalkers * niter
            flatchain = chain.reshape(ntot, nparams)
            lnp = lnprobability.reshape(ntot)
        else: # dynesty
            ntot, nparams = chain.shape
            flatchain = chain
            lnp = lnprobability
            
        theta = flatchain[lnp.argmax(), :] # maximum likelihood values
        print('Maximum likelihood values: ', theta)

        prob = np.exp(lnp - lnp.max())
        prob /= prob.sum()
        rand_indx = rand.choice(ntot, size=nrand, replace=False, p=prob)
        theta_rand = flatchain[rand_indx, :]

    print('Rendering the maximum-likelihood model...', end='')
    t0 = time.time()
    modelwave, modelspec, modelphot = _sed(model=model, theta=theta, obs=obs, sps=sps)
    print('...took {:.2f} sec'.format(time.time()-t0))
    #print(modelspec.min(), modelspec.max())

    # Establish the wavelength and flux limits.
    minwave, maxwave = 0.1, 35.0
    #minwave, maxwave = np.min(weff - 5*fwhm), np.max(weff + fwhm)

    inrange = (modelwave > minwave) * (modelwave < maxwave)
    maxflux = np.hstack( (galphot + 1.*galphoterr, modelspec[inrange]) ).max() + 0.5
    minflux = np.hstack( (galphot - 1.*galphoterr, modelspec[inrange]) ).min() - 0.5
    if maxflux > 30.:
        maxflux = 30.
    #minflux, maxflux = (12, 22)

    fig, ax = plt.subplots(figsize=(8, 6))
    if chain is not None and nrand > 0:
        for ii in range(nrand):
            _, r_modelspec, _ = _sed(model=model, theta=theta_rand[ii, :], obs=obs, sps=sps)
            ax.plot(modelwave, r_modelspec, alpha=0.8, color='gray')
    ax.plot(modelwave, modelspec, alpha=1.0, label='Model spectrum', color='k')
    
    ax.errorbar(weff, modelphot, marker='s', ls='', lw=3, markersize=15, markerfacecolor='none',
                markeredgewidth=3, alpha=0.6, label='Model photometry')
    ax.errorbar(weff[mask], galphot, yerr=galphoterr, marker='o', ls='', lw=2, markersize=10,
                markeredgewidth=2, alpha=0.8, label='Observed photometry',
                elinewidth=2, capsize=5)
                
    ax.set_xlabel(r'Observed-Frame Wavelength (${}$m)'.format('\mu'))
    ax.set_ylabel('AB mag')
    #ax.set_ylabel('Flux Density (mJy)')
    ax.set_xlim(minwave, maxwave)
    ax.set_ylim(minflux, maxflux)
    ax.set_xscale('log')
    ax.invert_yaxis()
    #ax.set_yscale('log')
    #ax.legend(loc='upper right', fontsize=16, frameon=True)
    # https://stackoverflow.com/questions/21920233/matplotlib-log-scale-tick-label-number-formatting
    ax.get_xaxis().set_major_formatter(FuncFormatter(lambda y, _: '{:.16g}'.format(y)))
    #ax.get_xaxis().set_major_formatter(ScalarFormatter())
    plt.subplots_adjust(left=0.15, right=0.95, bottom=0.18, top=0.95)

    # Add an inset with the posterior probability distribution.
    if False:
        ax1 = fig.add_axes([0.23, 0.68, 0.22, 0.22])
        ax1.hist(chain[:, 4], bins=50, histtype='step', linewidth=2, 
                 edgecolor='k',fill=True)    
        ax1.set_xlim(10.5, 11.5)
        ax1.set_yticklabels([])
        ax1.set_xlabel(r'$\log_{10}(\mathcal{M}/\mathcal{M}_{\odot})$')
        ax1.set_ylabel(r'$P(\mathcal{M})$')
        ax1.xaxis.set_major_locator(MultipleLocator(0.5))
        for item in ([ax1.title, ax1.xaxis.label, ax1.yaxis.label] +
                 ax1.get_xticklabels() + ax1.get_yticklabels()):
            item.set_fontsize(16)

    if png:
        print(f'Writing {png}')
        fig.savefig(png)

