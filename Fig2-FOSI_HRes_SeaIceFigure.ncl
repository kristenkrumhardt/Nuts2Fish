; Figure for FOSI HRes paper (Krumhardt et al)
; copied from /glade/u/home/landrum/CESM_CCSM4/AntIce/SeaBird_Predators/obs_nCESM_AICE_lon.ncl
; and modified specifically for use for Krumhardt et al CESM-FEISTY paper.
; LLandrum
; Sept 2023
;----------------------------------------------------------------------------------
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_code.ncl"
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_csm.ncl"
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/contributed.ncl"
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/shea_util.ncl"
;load "/home/landrum/ncl/landrum_util.ncl"
load "/glade/u/home/landrum/ncl/landrum_util.ncl"

begin
; ==============================================================
; User defined parameters that specify region of globe to consider
; ==============================================================
  HRcase = "g.e22.TL319_t13.G1850ECOIAF_JRA_HR.4p2z.001"
  HResdir = "/glade/campaign/cgd/ppc/landrum/"+HRcase+"/ice/"
  HRes_short = "TL319 JRA 4p2z"
  HRcase_yr1 = 1958
  HRcase_yr2 = 2021
  HRcase_preproc_yr1 = 1958
  HRcase_clim_yr1 = 1979
  HRcase_clim_yr2 = 2020

  syr1 =  sprinti("%0.4i", HRcase_yr1)   ; left zero fill
  syr2 =  sprinti("%0.4i", HRcase_yr2)   ; left zero fill

  HResfile = HResdir+"aice_sh_long_"+HRcase+"."+syr1+"01-"+syr2+"12.nc"
  HResfile_nh = HResdir+"pre_process/ice_area_ext_nh.MHolland_regions."+HRcase+"."+HRcase_preproc_yr1+"-"+HRcase_yr2+".nc"
  HResfile_sh = HResdir+"pre_process/ice_area_ext_sh."+HRcase+"."+HRcase_preproc_yr1+"-"+HRcase_yr2+".nc"
  HRes_clim = HResdir+HRcase+"."+HRcase_clim_yr1+"-"+HRcase_clim_yr2+".avg_clim_std.aice.nc"

  cesm_case = "g.e22.GOMIPECOIAF_JRA-1p4-2018.TL319_g17.4p2z.002branch"
  cesmdir = "/glade/work/landrum/projects/Antarctic_polynyas/JRA_4p2z/ice/"
  cesm_case_yr1 = 1958
  cesm_case_yr2 = 2021
  cesmfile_preproc_nh = cesmdir+"pre_process/ice_area_ext_nh.MHolland_regions."+cesm_case+"."+cesm_case_yr1+"-"+cesm_case_yr2+".nc"
  cesmfile_preproc_sh = cesmdir+"pre_process/ice_area_ext_sh."+cesm_case+"."+cesm_case_yr1+"-"+cesm_case_yr2+".nc"
  cesmfile = cesmdir+"aice_sh_long_"+cesm_case+"."+cesm_case_yr1+"01-"+cesm_case_yr2+"12.nc"
  LRes_clim = cesmdir+cesm_case+".avg_clim_std.aice."+HRcase_clim_yr1+"-"+HRcase_clim_yr2+".nc"

  detrend = True
  shade_STD = True				; True to shade std onto mean plot, false to show STD on separate plot
  smooth = True                                 ; True to smooth using running average over three longitude points
  plot_SH_long = False				; True to plot SIA, trends for SH as a function of longitude

  plot_yr1 = 1979				; yrs over which to calculate means and trends and then to plot
  plot_yr2 = 2020
  nyrs = plot_yr2 - plot_yr1+ 1
  nmnths = nyrs*12
  nyrs_trnds = nyrs

  ext_plot_yr1 = 1958                               ; yrs to plot SIE timeseries
  ext_plot_yr2 = 2021

  clim_ext_yr1 = 1979					; years over which to calculated climatologies
  clim_ext_yr2 = 2020

  var2plot = "ext"

  var_name_nh = var2plot+"_mo_nh"
  var_name_sh = var2plot+"_mo_sh"

  seasons = "ANN"
;  seasons = (/"ANN","JJA"/)
;  seasons = (/"ANN","DJF","AMJ","MAM","JJA","SON"/)
;  seasons = (/"DJF","MAM","SON"/)
  nseasons = dimsizes(seasons)
  season_labels = "Annual"

  month_names = (/"Jan","Feb","Mar","Apr","May","June","July","Aug","Sept","Oct","Nov","Dec"/)

  ssmi_yr1 = 1979
  ssmi_yr2 = 2020
  obs_type = "CDR"

  grid = "gx1v5"
  ssmidir = "/glade/p/cesm/pcwg/landrum/SSMI/"
;  ssmi_file = ssmidir+"/SSMI.ifrac."+ssmi_yr1+"01-"+ssmi_yr2+"12."+grid+".nc"                            ; this script assumes ssmi ifrac are on same grid as model
;  ssmi_file = "/glade/p/cesm/pcwg/ssmi/SH/ssmi_monthly_data_"+grid+"_"+ssmi_yr1+"-"+ssmi_yr2+".nc"				; this script assumes ssmi ifrac are on same grid as model
  ssmi_long_file = ssmidir+"/SH/ifrac_sh_long_SSMI_CDR."+ssmi_yr1+"01-"+ssmi_yr2+"12.nc"                              ; this script assumes ssmi ifrac are on same grid as model

  nsidc_ext_dir_nh = "/glade/p/cesm/pcwg/mholland/ssmi_ice_index/NH/csv/N"
  nsidc_ext_dir_sh = "/glade/p/cesm/pcwg/mholland/ssmi_ice_index/SH/csv/S"

  fssmi = ssmidir+"SH/ssmi_cdr_clim.SH."+ssmi_yr1+"-"+ssmi_yr2+".nc"
  ssmi_source = "CDR"

  fssmiNH = ssmidir+"ssmi_clim.NH."+ssmi_yr1+"-2014.nc"
  ssmi_sourceNH = "NasaTeam"

  cesm_grid_case = "b.e11.B1850C5CN.f09_g16.005"
  fgrid = "/glade/p/cesm/pcwg/landrum/"+cesm_grid_case+"/"+cesm_grid_case+".pop.h.SST.190001-199912.nc"			; needs to be ocn file so no missing TAREA. Here I am taking cesmL
;  fgrid = "/glade/p/cesm/omwg/grids/"+grid+"_grid.nc"
  
  psout = get_script_prefix_name() + "."+var2plot

  if (plot_SH_long) then
     if (detrend)
        psout = psout+".dtrnd"
     end if
     if (shade_STD) then
        psout = psout+".ShadeSTD"
     end if
     if (smooth) then
        psout = psout+".3pnt_runavg"
     end if
  end if
  psout_season = new(nseasons,typeof(psout))
; ==============================================================
; Antarctic and Arctic climatologies...
;--------------------------------------------------------------------------
  f1      = addfile (HRes_clim, "r")
  tmpHRaice = f1->aice_clim
  f2      = addfile (LRes_clim, "r")
  tmpLRaice = f2->aice_clim

  if (max(tmpHRaice).lt.1.1) then
     tmpHRaice = 100.*tmpHRaice
  end if
  if (max(tmpLRaice).lt.1.1) then
     tmpLRaice = 100.*tmpLRaice
  end if

  HR_tlat = f1->tlat
  HR_tlon = f1->tlon
  dims = dimsizes(tmpHRaice)
  nj = dims(1)
  ni = dims(2)
  HR_aice_mn = new((/nseasons,nj,ni/),typeof(tmpHRaice))
  HR_aice_mn!0 = "season"
  HR_aice_mn@lat2d = HR_tlat
  HR_aice_mn@lon2d = HR_tlon
  delete(f1)

  LR_tlat = f2->tlat
  LR_tlon = f2->tlon
  dims = dimsizes(tmpLRaice)
  LRnj = dims(1)
  LRni = dims(2)
  LR_aice_mn = new((/nseasons,LRnj,LRni/),typeof(tmpLRaice))
  LR_aice_mn!0 = "season"
  LR_aice_mn@lat2d = LR_tlat
  LR_aice_mn@lon2d = LR_tlon
  delete(f2)

  f3      = addfile (fssmi, "r")
  tmp_ifrac_sh = f3->CDR_clim
  if (max(tmp_ifrac_sh).lt.1.1) then
     tmp_ifrac_sh = 100.*tmp_ifrac_sh                                         ; ssmi data is 0-1.
  end if
  lat_obs = f3->lat
  lon_obs = f3->lon

  ssmi_mn_sh = tmp_ifrac_sh(:nseasons-1,:,:)
  ssmi_mn_sh@lat2d = lat_obs
  ssmi_mn_sh@lon2d = lon_obs
  delete(f3)

  f4      = addfile (fssmiNH, "r")					; no CDR yet....
  tmp_ifrac_nh = f4->Nasa_Team_clim
  if (max(tmp_ifrac_nh).lt.1.1) then
     tmp_ifrac_nh = 100.*tmp_ifrac_nh                                              ; ssmi data is 0-1.
  end if
  latNH_obs = f4->lat
  lonNH_obs = f4->lon

  ssmi_mn_nh = tmp_ifrac_nh(:nseasons-1,:,:)
  ssmi_mn_nh@lat2d = latNH_obs
  ssmi_mn_nh@lon2d = lonNH_obs

  lat3d = conform(tmp_ifrac_nh,latNH_obs,(/1,2/))
  tmp_ifrac_nh = where(lat3d.ge.85.,tmp_ifrac_nh@_FillValue,tmp_ifrac_nh)                   ; pole hole in earlier satellite data

  do is=0,nseasons-1
     if (typeof(seasons).eq."string") then
        season = seasons(is)
        if (season.ne."ANN") then
           tmp_HRseasonal = month_to_season(tmpHRaice,season)
           tmp_LRseasonal = month_to_season(tmpLRaice,season)
           tmp_SHseasonal = month_to_season(tmp_ifrac_sh,season)
           tmp_NHseasonal = month_to_season(tmp_ifrac_nh,season)
        else
           tmp_HRseasonal = month_to_annual(tmpHRaice,1)
           tmp_LRseasonal = month_to_annual(tmpLRaice,1)
           tmp_SHseasonal = month_to_annual(tmp_ifrac_sh,1)
           tmp_NHseasonal = month_to_annual(tmp_ifrac_nh,1)
        end if
     else
        tmp_HRseasonal = tmpHRaice(seasons(is)-1,:,:)
        tmp_LRseasonal = tmpLRaice(seasons(is)-1,:,:)
        tmp_SHseasonal = tmp_ifrac_sh(seasons(is)-1::12,:,:)
        tmp_NHseasonal = tmp_ifrac_nh(seasons(is)-1::12,:,:)
     end if

     LR_aice_mn(is,:,:) = (/ tmp_LRseasonal /)
     HR_aice_mn(is,:,:) = (/ tmp_HRseasonal /)
     ssmi_mn_sh(is,:,:) = (/ tmp_SHseasonal /)
     ssmi_mn_nh(is,:,:) = (/ tmp_NHseasonal /)

     delete(tmp_SHseasonal)
     delete(tmp_HRseasonal)
     delete(tmp_LRseasonal)
     delete(tmp_NHseasonal)
  end do
  delete(tmpHRaice)
  delete(tmpLRaice)
  delete(tmp_ifrac_sh)
  delete(tmp_ifrac_nh)

;  HR_aice_mn = mask(HR_aice_mn,HR_aice_mn.gt.1.,True)                   ; (plotting purposes) mask out areas of seasonally averaged aice < 1%
  HR_aice_mn = mask(HR_aice_mn,HR_aice_mn.gt.15.,True)                   ; (plotting purposes) mask out areas of seasonally averaged aice < 15%

  LR_aice_mn = mask(LR_aice_mn,LR_aice_mn.gt.1.,True)                   ; (plotting purposes) mask out areas of seasonally averaged aice < 1%

;--------------------------------------------------------------------------
; SH and NH SIE timeseries
;--------------------------------------------------------------------------
  ext_year = ispan(ext_plot_yr1,ext_plot_yr2,1)
  ext_nyr_plot = dimsizes(ext_year)

  sie_mo_nh = new((/3,12*ext_nyr_plot/),float)
  sie_seasonal_nh = new((/3,nseasons,ext_nyr_plot/),float)

  sie_mo_sh = new((/3,12*ext_nyr_plot/),float)
  sie_seasonal_sh = new((/3,nseasons,ext_nyr_plot/),float)

  sie_clim_nh = new((/3,12/),float)
  sie_clim_sh = new((/3,12/),float)
; ----------------HRes FOSI-------------------------------------------------------

  if (ext_plot_yr1.ge.HRcase_preproc_yr1) then
     i1 = (ext_plot_yr1 - HRcase_yr1)*12
     ii1 = 0
  else
     i1 = 0
     ii1 = (HRcase_preproc_yr1 - ext_plot_yr1)*12
  end if
  if (ext_plot_yr2.le.HRcase_yr2) then
     i2 = (ext_plot_yr2 - HRcase_preproc_yr1 + 1)*12 - 1
     ii2 = (ext_plot_yr2 - ext_plot_yr1 + 1)*12 -1
  else
     i2 = (HRcase_yr2 - HRcase_preproc_yr1 + 1)*12 - 1
     ii2 = (HRcase_yr2 - ext_plot_yr1 + 1)*12 -1
  end if
  f      = addfile (HResfile_nh, "r")
  sie_mo_nh(0,ii1:ii2) = (/ f->$var_name_nh$(i1:i2) /)
  delete(f)

  f      = addfile (HResfile_sh, "r")
  sie_mo_sh(0,ii1:ii2) = (/ f->$var_name_sh$(i1:i2) /)

;
; ----------------LRes FOSI-------------------------------------------------------

  if (ext_plot_yr1.ge.cesm_case_yr1) then
     i1 = (ext_plot_yr1 - cesm_case_yr1)*12
     ii1 = 0
  else
     i1 = 0
     ii1 = (cesm_case_yr1 - ext_plot_yr1)*12
  end if
  if (ext_plot_yr2.le.cesm_case_yr2) then
     i2 = (ext_plot_yr2 - cesm_case_yr1 + 1)*12 - 1
     ii2 = (ext_plot_yr2 - ext_plot_yr1 + 1)*12 -1
  else
     i2 = (cesm_case_yr2 - cesm_case_yr1 + 1)*12 - 1
     ii2 = (cesm_case_yr1 - ext_plot_yr1 + 1)*12 -1
  end if
  f2      = addfile (cesmfile_preproc_nh, "r")
  sie_mo_nh(1,ii1:ii2) = (/ f2->$var_name_nh$(i1:i2) /)
  delete(f2)

  f2      = addfile (cesmfile_preproc_sh, "r")
  sie_mo_sh(1,ii1:ii2) = (/ f2->$var_name_sh$(i1:i2) /)
  delete(f2)

; ----------------ssmi-------------------------------------------------------

  if (var2plot.eq."ext") then
     ivar = 4
  else
     ivar = 5
  end if

  do ihem=0,1
     if (ihem.eq.0) then
        nsidc_ext_dir = nsidc_ext_dir_nh
     else
        nsidc_ext_dir = nsidc_ext_dir_sh
     end if 

     do i=1,12
        smn = sprinti("%0.2i",i)
        fnsidc = nsidc_ext_dir+"_"+smn+"_extent_v3.0.csv"
        extdata_str = asciiread(fnsidc,-1,"string")
        tmp1 = str_split_csv(extdata_str,",",0)
        nsidc_yr = toint(tmp1(1:,0))
        nsidc_yr1 = nsidc_yr(0)
        nyr_nsidc = dimsizes(nsidc_yr)
        nsidc_yr2 = nsidc_yr(nyr_nsidc-1)

        if (ext_plot_yr1.ge.nsidc_yr1) then
           i1 = (ext_plot_yr1 - nsidc_yr1)
           ii1 = 0
        else
           i1 = 0
           ii1 = (nsidc_yr1 - ext_plot_yr1)*12
        end if

        if (ext_plot_yr2.le.nsidc_yr2) then
           i2 = (ext_plot_yr2 - nsidc_yr1)
           ii2 = (ext_plot_yr2 - ext_plot_yr1)*12
        else
           i2 = (nsidc_yr2 - nsidc_yr1)
           ii2 = (nsidc_yr2 - ext_plot_yr1)*12
        end if
        im1 = ii1 + i - 1
        im2 = ii2 + i - 1
        if (ihem.eq.0) then
           sie_mo_nh(2,im1:im2:12) = (/ tofloat(tmp1(1+i1:1+i2,ivar)) /)
        else
           sie_mo_sh(2,im1:im2:12) = (/ tofloat(tmp1(1+i1:1+i2,ivar)) /)
        end if

        delete(extdata_str)
        delete(tmp1)
        delete(nsidc_yr)
     end do
  end do

  sie_mo_nh(2,:) = where(sie_mo_nh(2,:).lt.0.,sie_mo_nh@_FillValue,sie_mo_nh(2,:))
  sie_mo_sh(2,:) = where(sie_mo_sh(2,:).lt.0.,sie_mo_sh@_FillValue,sie_mo_sh(2,:))

  sie_mo_nh!1 = "time"
  sie_mo_sh!1 = "time"

  do is=0,nseasons-1
     season = seasons(is)
     if (season.ne."ANN") then
        do i=0,2
           sie_seasonal_sh(i,is,:) = (/ month_to_season(sie_mo_sh(i,:),season) /)
           sie_seasonal_nh(i,is,:) = (/ month_to_season(sie_mo_nh(i,:),season) /)
        end do
     else
        do i=0,2
           sie_seasonal_sh(i,is,:) = (/ month_to_annual(sie_mo_sh(i,:),1) /)
           sie_seasonal_nh(i,is,:) = (/ month_to_annual(sie_mo_nh(i,:),1) /)
        end do
     end if
  end do

  ext_plot_yr1 = 1958                               ; yrs to plot SIE timeseries
  ext_plot_yr2 = 2021

  clim_ext_yr1 = 1979                                   ; years over which to calculated climatologies
  clim_ext_yr2 = 2020

  im1 = (clim_ext_yr1 - ext_plot_yr1)*12
  im2 = (clim_ext_yr2 - ext_plot_yr1 +1)*12 - 1
  tmp_sie_mo_nh = sie_mo_nh(:,im1:im2)
  tmp_sie_mo_sh = sie_mo_sh(:,im1:im2)

  do im=0,11
     sie_clim_nh(:,im) = (/ dim_avg_n(tmp_sie_mo_nh(:,im::12),1) /)
     sie_clim_sh(:,im) = (/ dim_avg_n(tmp_sie_mo_sh(:,im::12),1) /)
  end do
  delete(tmp_sie_mo_nh)
  delete(tmp_sie_mo_sh)

  sie_clim_nh!0 = "dataset"
  sie_clim_nh!1 = "month"
  sie_clim_nh&month = ispan(1,12,1)
  sie_clim_sh!0 = "dataset"
  sie_clim_sh!1 = "month"
  sie_clim_sh&month = ispan(1,12,1)

  sie_seasonal_nh!0 = "dataset"
  sie_seasonal_nh!1 = "season"
  sie_seasonal_nh!2 = "year"
  sie_seasonal_nh&year = ext_year

  sie_seasonal_sh!0 = "dataset"
  sie_seasonal_sh!1 = "season"
  sie_seasonal_sh!2 = "year"
  sie_seasonal_sh&year = ext_year

;--------------------------------------------------------------------------
; Antarctic SIA as a function of longitude...
;--------------------------------------------------------------------------
  year = ispan(plot_yr1,plot_yr2,1)

;  print("Calculating trends for "+HRcase)
  i1 = (plot_yr1 - HRcase_yr1)*12
  i2 = (plot_yr2 - HRcase_yr1 + 1)*12 - 1
  f      = addfile (HResfile, "r")
  tmpaice = f->aice_sh_long(i1:i2,:)
  HRes_lon1d = f->longitude

; ----------------need to normalize for comparing with data - in this case obs - that are on another grid...
;                  calculate as m^2 per degree longitude
  dHRlon_for = HRes_lon1d
  nHR = dimsizes(HRes_lon1d)

  dHRlon_for(:nHR-2) = HRes_lon1d(1:) - HRes_lon1d(:nHR-2)
  dHRlon_for(nHR-1) = HRes_lon1d(0) - HRes_lon1d(nHR-1)
  dHRlon_for = where(dHRlon_for.lt.0.,dHRlon_for+360.,dHRlon_for)

  dHRlon_back = HRes_lon1d
  dHRlon_back(1:nHR-1) = HRes_lon1d(1:) - HRes_lon1d(:nHR-2)
  dHRlon_back(0) = HRes_lon1d(0) - HRes_lon1d(nHR-1)
  dHRlon_back = where(dHRlon_back.lt.0.,dHRlon_back+360.,dHRlon_back)
 
  dHR = dHRlon_for
  dHR = 0.5*(dHRlon_for+dHRlon_back)			; use for normalizing data
  dHR2d = conform(tmpaice,doubletofloat(dHR),1)

  tmpaice = tmpaice/dHR2d

  if (HRes_lon1d(nHR-1).lt.HRes_lon1d(0)) then		; not monotonically increasing
     dlon = HRes_lon1d
     dlon(1:) = dlon(1:) - dlon(:nHR-2)
     ilon = min(ind(dlon.lt.0.))
     if (HRes_lon1d(ilon).gt.0.) then
        HRes_lon1d(0:ilon-1) = HRes_lon1d(0:ilon-1) - 360.
     else
        HRes_lon1d(ilon:) = HRes_lon1d(ilon:) + 360.
     end if
     delete(dlon)
  end if

  HRes_aice_long = new((/nyrs,nseasons,nHR/),typeof(tmpaice))
  HRes_aice_long!0 = "year"
  HRes_aice_long!1 = "season"
  HRes_aice_long!2 = "longitude"
  HRes_aice_long&longitude = HRes_lon1d

  HRes_aice_long_trnd = new((/nseasons,nHR/),typeof(tmpaice))
  HRes_aice_long_trnd!0 = "season"
  HRes_aice_long_trnd!1 = "longitude"
  HRes_aice_long_trnd&longitude = HRes_lon1d

  do is=0,nseasons-1
     season = seasons(is)
     if (season.ne."ANN") then
        tmp3d = new((/nmnths,2,nHR/),typeof(tmpaice))
        tmp3d(:,0,:) = tmpaice
        tmp3d(:,1,:) = tmpaice

        tmp_seasonal = month_to_season(tmp3d,season)
        HRes_aice_long(:,is,:) = (/ tmp_seasonal(:,0,:) /)

        delete(tmp3d)
        delete(tmp_seasonal)
     else
        HRes_aice_long(:,is,:) = (/ month_to_annual(tmpaice,1) /)
     end if

     aiceDtrnd = dtrend_msg_n(year,HRes_aice_long(:,is,:),True,True,0)
     HRes_aice_long_trnd(is,:) = aiceDtrnd@slope
  end do
  delete(tmpaice)
  delete(aiceDtrnd)

  mn_HRes_aice_long = dim_avg_n_Wrap(HRes_aice_long,0)
 
  if (detrend)
     tmp = dtrend_msg_n(year,HRes_aice_long,True,False,0)
     std_HRes_aice_long = dim_stddev_n_Wrap(tmp,0)
     delete(tmp)
     std_HRes_aice_long!0 = "season"
     std_HRes_aice_long!1 = "longitude"
     std_HRes_aice_long&longitude = HRes_lon1d
  else
     std_HRes_aice_long = dim_stddev_n_Wrap(HRes_aice_long,0)
  end if
  delete(HRes_aice_long)
;-------------------------------------------------------------------------------------------
; grid info for normalizing. obs, cesm 
;------------------------------------------------------------------------------------------

  f = addfile(fgrid,"r")
  tarea = doubletofloat(f->TAREA(:75,:))
  tarea = tarea/(100.*100.)					; cm2 to m2
  tlat = doubletofloat(f->TLAT(:75,:))
  tlon = doubletofloat(f->TLONG(:75,:))

  n10 = max(ind(tlat(:,0).lt.-10.))
  tlon1d = dim_avg_n(tlon(:n10,:),0)
  nlon = dimsizes(tlon1d)
  if (tlon1d(nlon-1).lt.tlon1d(0)) then			; not monotonically increasing
     dlon = tlon1d
     dlon(1:) = dlon(1:) - dlon(:nlon-2)
     ilon = min(ind(dlon.lt.0.))
     if (tlon1d(ilon).gt.0.) then
        tlon1d(0:ilon-1) = tlon1d(0:ilon-1) - 360.
     else
        tlon1d(ilon:) = tlon1d(ilon:) + 360.
     end if
     delete(dlon)
  end if

  dlon_for = tlon1d
  dlon_for(:nlon-2) = tlon1d(1:) - tlon1d(:nlon-2)
  dlon_for(nlon-1) = tlon1d(0) - tlon1d(nlon-1)
  dlon_for = where(dlon_for.lt.0.,dlon_for+360.,dlon_for)

  dlon_back = tlon1d
  dlon_back(1:nlon-1) = tlon1d(1:) - tlon1d(:nlon-2)
  dlon_back(0) = tlon1d(0) - tlon1d(nlon-1)
  dlon_back = where(dlon_back.lt.0.,dlon_back+360.,dlon_back)

  dcesm = dlon_for
  dcesm = 0.5*(dlon_for+dlon_back)                    ; use for normalizing data

;-------------------------------------------------------------------------------------------
; JRA hindcast ....
;------------------------------------------------------------------------------------------
  i1 = (plot_yr1 - cesm_case_yr1)*12
  i2 = (plot_yr2 - cesm_case_yr1 + 1)*12 - 1
  f      = addfile (cesmfile, "r")
  tmpaice = f->aice_sh_long(i1:i2,:)

  dcesm2d = conform(tmpaice,dcesm,1)
  tmpaice = tmpaice/dcesm2d

  JRA_aice_long = new((/nyrs,nseasons,nlon/),typeof(tmpaice))
  JRA_aice_long!0 = "year"
  JRA_aice_long!1 = "season"
  JRA_aice_long!2 = "longitude"
  JRA_aice_long&longitude = tlon1d

  JRA_aice_long_trnd = new((/nseasons,nlon/),typeof(tmpaice))
  JRA_aice_long_trnd!0 = "season"
  JRA_aice_long_trnd!1 = "longitude"
  JRA_aice_long_trnd&longitude = tlon1d

  do is=0,nseasons-1
     season = seasons(is)
     if (season.ne."ANN") then
        tmp3d = new((/nmnths,2,nlon/),typeof(tmpaice))
        tmp3d(:,0,:) = tmpaice
        tmp3d(:,1,:) = tmpaice

        tmp_seasonal = month_to_season(tmp3d,season)
        JRA_aice_long(:,is,:) = (/ tmp_seasonal(:,0,:) /)

        delete(tmp3d)
        delete(tmp_seasonal)
     else
        JRA_aice_long(:,is,:) = (/ month_to_annual(tmpaice,1) /)
     end if

     aiceDtrnd = dtrend_msg_n(year,JRA_aice_long(:,is,:),True,True,0)
     JRA_aice_long_trnd(is,:) = aiceDtrnd@slope
  end do
  delete(tmpaice)
  delete(aiceDtrnd)

  mn_JRA_aice_long = dim_avg_n_Wrap(JRA_aice_long,0)

  if (detrend)
     tmp = dtrend_msg_n(year,JRA_aice_long,True,False,0)
     std_JRA_aice_long = dim_stddev_n_Wrap(tmp,0)
     delete(tmp)
     std_JRA_aice_long!0 = "season"
     std_JRA_aice_long!1 = "longitude"
     std_JRA_aice_long&longitude = tlon1d
  else
     std_JRA_aice_long = dim_stddev_n_Wrap(JRA_aice_long,0)
  end if
  delete(JRA_aice_long)
;-------------------------------------------------------------------------------------------
; now observations....
;------------------------------------------------------------------------------------------

  ifrac_long = new((/nyrs_trnds,nseasons,nlon/),float)
  ifrac_long!0 = "year"
  ifrac_long!1 = "season"
  ifrac_long!2 = "longitude"
  ifrac_long&longitude = tlon1d

  ifrac_long_trnd = new((/nseasons,nlon/),float)
  ifrac_long_trnd!0 = "season"
  ifrac_long_trnd!1 = "longitude"
  ifrac_long_trnd&longitude = tlon1d

  i1 = (plot_yr1 - ssmi_yr1)*12
  i2 = (plot_yr2 - ssmi_yr1 + 1)*12 - 1
  if (obs_type.eq."CDR") then						; ifrac as a function of longitude already exists
     f = addfile(ssmi_long_file,"r")
     tmp_ifrac_long_CDR = f->ifrac_sh_long(i1:i2,:)
     tmp3d = new((/nyrs_trnds*12,nlon,1/),typeof (tmp_ifrac_long_CDR))       ; need 3d for month_to_season
     tmp3d(:,:,0) = tmp_ifrac_long_CDR
     do is=0,nseasons-1
        season = seasons(is)
        if (season.ne."ANN") then
           tmp_season = month_to_season(tmp3d,season)
           tmp_ifrac_long = tmp_season(:,:,0)
           delete(tmp_season)
        else
           tmp_ifrac_long = month_to_annual(tmp_ifrac_long_CDR,1)
        end if

        if (is.eq.0) then
            dobs2d = conform(tmp_ifrac_long,dcesm,1)
        end if
        tmp_ifrac_long = tmp_ifrac_long/dobs2d
        ifrac_long(:,is,:) = (/ tmp_ifrac_long /)                                  ; aice in m2 as fnctn of longitude

        ifracDtrnd = dtrend_msg_n(year,tmp_ifrac_long,True,True,0)
        ifrac_long_trnd(is,:) = ifracDtrnd@slope

        delete(tmp_ifrac_long)
        delete(ifracDtrnd)
     end do
     delete(tmp_ifrac_long_CDR)
     delete(tmp3d)
  else
     f = addfile(ssmi_file,"r")
     if (obs_type.eq."Bootstrap") then
        tmp_ifrac = f->Bootstrap(i1:i2,:75,:)                                 ; only read in 1st 76 tlat for SH
     else
        tmp_ifrac = f->Nasa_Team(i1:i2,:75,:)                                 ; only read in 1st 76 tlat for SH
     end if
     tmp_ifrac@_FillValue = 9.96921e+36                                           ; SSMI uses 254 for land
     tmp_ifrac = where(tmp_ifrac.ge.200, tmp_ifrac@_FillValue,tmp_ifrac)
;     tmp_ifrac = where(tmp_ifrac.eq.-999, tmp_ifrac@_FillValue,tmp_ifrac)
     tmp_ifrac = where(tmp_ifrac.lt.0, tmp_ifrac@_FillValue,tmp_ifrac)
     tmp_ifrac = 0.01*tmp_ifrac					; from % to concentration

     tarea3d = conform(tmp_ifrac(:nyrs-1,:,:),tarea,(/1,2/))

     do is=0,nseasons-1
        season = seasons(is)
        if (season.ne."ANN") then
           tmp_seasonal = month_to_season(tmp_ifrac,season)
        else
           tmp_seasonal = month_to_annual(tmp_ifrac,1)
        end if

        tmp_ifrac_m2 = tmp_seasonal*tarea3d*1.e-10                                  ; convert ice area from % to m^2 for averaging over latitude
        tmp_ifrac_long = dim_sum_n(tmp_ifrac_m2,1)
        if (is.eq.0) then
            dobs2d = conform(tmp_ifrac_long,dcesm,1)
        end if
        tmp_ifrac_long = tmp_ifrac_long/dobs2d
        ifrac_long(:,is,:) = (/ tmp_ifrac_long /)                                  ; aice in m2 as fnctn of longitude

        ifracDtrnd = dtrend_msg_n(year,tmp_ifrac_long,True,True,0)
        ifrac_long_trnd(is,:) = ifracDtrnd@slope

        delete(tmp_ifrac_long)
        delete(tmp_seasonal)
        delete(ifracDtrnd)
        delete(tmp_ifrac_m2)
     end do
     delete(tmp_ifrac)
  end if

  mn_ifrac_long = dim_avg_n_Wrap(ifrac_long,0)
  if (detrend)
     tmp = dtrend_msg_n(year,ifrac_long,True,False,0)
     std_ifrac_long = dim_stddev_n_Wrap(tmp,0)
     delete(tmp)
     std_ifrac_long!0 = "season"
     std_ifrac_long!1 = "longitude"
     std_ifrac_long&longitude = tlon1d
  else
     std_ifrac_long = dim_stddev_n_Wrap(ifrac_long,0)
  end if

  delete(ifrac_long)
;-------------------------------------------------------------------------------------------
; pivot about 0 deg east for plotting
;------------------------------------------------------------------------------------------
;  decor_pivot = lonPivot(decorrel_lon,1.)

;-------pivot about 0 deg east for plotting if longitude does not start ~0
  if (min(HRes_lon1d).lt.0.) then                   ; pivot
     mn_HRes_aice_long = lonPivot(mn_HRes_aice_long,1.)
     std_HRes_aice_long = lonPivot(std_HRes_aice_long,1.)
     HRes_aice_long_trnd = lonPivot(HRes_aice_long_trnd,1.)
  end if

  if (min(tlon1d).lt.0.) then
     mn_JRA_aice_long = lonPivot(mn_JRA_aice_long,1.)
     std_JRA_aice_long = lonPivot(std_JRA_aice_long,1.)
     JRA_aice_long_trnd = lonPivot(JRA_aice_long_trnd,1.)

     mn_ifrac_long = lonPivot(mn_ifrac_long,1.)
     std_ifrac_long = lonPivot(std_ifrac_long,1.)
     ifrac_long_trnd = lonPivot(ifrac_long_trnd,1.)
  end if
;-------------------------------------------------------------------------------------------
; multiply trends by 100 to make for neater labels on y axis
;------------------------------------------------------------------------------------------
  HRes_aice_long_trnd = 100.*HRes_aice_long_trnd
  JRA_aice_long_trnd = 100.*JRA_aice_long_trnd
  ifrac_long_trnd = 100.*ifrac_long_trnd
;-------------------------------------------------------------------------------------------------------------
; smooth: smoothing over 3 deg long - HRes tends to be 0.1 deg so 10x number of points for smoothing as others
;-------------------------------------------------------------------------------------------------------------
  if (smooth) then
     opt = -1                              ; use cyclic conditions in running average....
     mn_ifrac_long = (/ runave_n(mn_ifrac_long,3,opt,1) /)
     std_ifrac_long = (/ runave_n(std_ifrac_long,3,opt,1) /)
     ifrac_long_trnd = (/ runave_n(ifrac_long_trnd,3,opt,1) /)

     mn_JRA_aice_long = (/ runave_n(mn_JRA_aice_long,3,opt,1) /)
     std_JRA_aice_long = (/ runave_n(std_JRA_aice_long,3,opt,1) /)
     JRA_aice_long_trnd = (/ runave_n(JRA_aice_long_trnd,3,opt,1) /)

     mn_HRes_aice_long = (/ runave_n(mn_HRes_aice_long,30,opt,1) /)
     std_HRes_aice_long = (/ runave_n(std_HRes_aice_long,30,opt,1) /)
     HRes_aice_long_trnd = (/ runave_n(HRes_aice_long_trnd,30,opt,1) /)
  end if
;************************************************
; resource list for contours
;************************************************
  aice_Pcntrs=(/1.,5.,10.,15.,20.,30.,40.,50., 60., 70., 80.,90.,95., 99./) ; aice
  line_colors = (/"black","blue","red"/)
  colormaps = (/"BkBlAqGrYeOrReViWh200","cmocean_ice"/)

  res                      = True                 ; plot mods desired
  res@gsnDraw		   = False		
  res@gsnFrame		   = False		
;  res@gsnStringFontHeightF = 0.015
  res@gsnStringFontHeightF = 0.03
  res@gsnMaximize           = False         ; enlarge plot
  res@gsnCenterString = ""
  res@gsnLeftString = ""

  resc = True				; contour map resources
  resc = res

  resc@cnFillPalette          = colormaps(1)
;  res@cnFillPalette          = colormaps(0)

  res@vpHeightF= 0.4                    ; change aspect ratio of plot
  res@vpWidthF = 0.8
  res_ext = True                                ; SIE ts plot resources
  res_ext = res

;  res@tiYAxisString= "Sea Ice Area (10~S~10~N~ m~S~2~N~)"
;  res@tiYAxisString= "Sea Ice Area (10~S~4~N~ km~S~2~N~)"
  res@tiXAxisString= "Longitude"

  res_ext@tiYAxisString= "10~S~6~N~ km~S~2~N~"
  res_ext@tiXAxisString= "Year"

  resc@cnFillOn             = True                 ; turn on color fill
  resc@lbLabelAutoStride    = True                 ; control labelbar labels
  resc@cnLinesOn            = False                ; turn off contour lines
  resc@cnLineLabelsOn       = False              ; turn the line labels off

  resc@gsnSpreadColors      = True                 ; use full colormap
  resc@lbLabelBarOn        = False                 ; Turn off labelbar

  resc@cnInfoLabelOn       = False       ; Turn off informational label

  resc@gsnSpreadColors      = True                 ; use full colormap
  res@xyMonoLineThickness = True
  res@xyMonoDashPattern = True

  resc@gsnLeftStringOrthogonalPosF = 0.03
  resc@trGridType = "TriangularMesh"

  resc@gsnPolar  = "SH"
  resc@mpMaxLatF =   -60.
  resc@gsnAddCyclic = True
  resc@cnLevels             =  aice_Pcntrs
  resc@cnLevelSelectionMode = "ExplicitLevels" ; set manual contour levels
  resc@cnFillColors = (/27,43,59,75,91,107,123,139,155,171,187,203,219,235,251/)                         ; cmocean_ice
  resc@gsnPolarLabelFontHeightF = 0.015
;  resc@gsnRightString = "%"
;  resc@gsnCenterString = HRcase_clim_yr1+"-"+HRcase_clim_yr2+" Mean"
  resc@gsnCenterString = ""

  res_ext@xyLineThicknessF = 2.0
  res_ext@xyLineColors = line_colors
  res_ext@xyMonoDashPattern = True
  res_ext@xyMonoLineThickness = True
;  tmsize = 0.04
;  res_ext@tmXBLabelFontHeightF = tmsize
;  res_ext@tmYLLabelFontHeightF = tmsize 
;  res_ext@tmYRLabelFontHeightF = tmsize 


  res_clim = True
  res_clim = res_ext
  res_clim@tmXBMode          = "Explicit"              ; explicit labels
  res_clim@tmXBValues        = ispan(1,12,1)
  res_clim@tmXBLabels        = (/"J","F","M","A","M","J","J","A","S","O","N","D"/)
  res_clim@tiXAxisString= "Month"
  res_clim@trXMinF = 1
  res_clim@trXMaxF = 12

  res_ext@trXMinF = ext_plot_yr1
  res_ext@trXMaxF = ext_plot_yr2

  pres = True
  pres@gsnFrame = False
  pres@txFontHeightF = 0.01
  pres@gsnMaximize     = True                 ; fill up the page
  pres@gsnPanelLabelBar  = True
  pres@lbTitleOn = True
  pres@lbTitleString = "%"
  pres@lbTitleFontHeightF = 0.01
;  pres@lbTitlePosition = "Right"
;  pres@lbTitleDirection = "Across"
  pres@lbLabelAutoStride = True
  pres@gsnMaximize     = True                 ; fill up the page

   xHRes = (/ mn_HRes_aice_long&longitude /)
   xp_HR = new(2*nHR,float)
   xp_HR(:nHR-1) = xHRes
   xp_HR(nHR:) = xHRes(::-1)
   yp_HR = new(2*nHR,float)
   xSSMI = (/ mn_ifrac_long&longitude /)
   xp = new(2*nlon,float)
   xp(:nlon-1) = xSSMI
   xp(nlon:) = xSSMI(::-1)
   yp = new(2*nlon,float)

   res@tmXBMode          = "Explicit"              ; explicit labels
   res@tmXBValues        = (/60,120,180,240,300/)  ; location of labels 
   res@tmXBLabels        = (/"60E","120E","180","120W","60W"/); labels themselves
   res@tmLabelAutoStride = True                    ; nice stride on labels
   res@trXMaxF = 360.

   res@xyLineThicknessF = 3.0

   res_ssmi = True
   res_ssmi = res
   res_JRA = True
   res_JRA = res

   res@xyLineColor = line_colors(0)
   res_JRA@xyLineColor = line_colors(1)
   res_ssmi@xyLineColor = line_colors(2)

  lres = True
  lres@gsLineDashPattern = 0

; legend resources
  lgres                    = True
  lgres@lgMonoLineColor    = False
  lgres@lgLineThicknessF   = 2
;  lgres@lgMonoDashIndex = True
;  lgres@lgDashIndex        = 0
  lgres@lgMonoDashIndex = False
  lgres@lgDashIndexes        = (/0,0,0,2/)
;  lgres@lgLabelFontHeightF = .018            ; legend label font thickness
  lgres@lgLabelFontHeightF = .025            ; legend label font thickness
  lgres@vpWidthF           = 0.18           ; width of legend (NDC)
  lgres@vpHeightF          = 0.04           ; height of legend (NDC)
  lgres@lgPerimOn = False

   resM = True
   resM = True
   resM@gsnDraw = False
   resM@gsnFrame = False
   resM@mpMinLatF = -80.
   resM@mpMaxLatF = -40.
   resM@mpCenterLonF =  180.
   resM@tmXBMode          = "Explicit"              ; explicit labels
   resM@tmXBValues        = (/60,120,180,240,300/)  ; location of labels
   resM@tmXBLabels        = (/"60E","120E","180","120W","60W"/); labels themselves
   res@tmLabelAutoStride = True                    ; nice stride on labels

   resM@vpWidthF = 0.8
   resM@vpHeightF = 0.15
;   resM@vpYF = 0.4
;   resM@vpXF = 0.125
   resM@tmXBLabelFontHeightF = 0.015
   resM@tmYROn = False
   resM@tmYLOn = False

   res15per = True
   res15per@gsnDraw          = False     ; don't draw
   res15per@gsnFrame         = False     ; don't advance frame
   res15per@cnLevelSelectionMode =  "ExplicitLevels"
   res15per@cnInfoLabelOn    =  False
   res15per@gsnLeftString    =  ""
   res15per@cnLevels         =  15.
   res15per@tiXAxisString    = ""
   res15per@tiYAxisString    = ""
   res15per@cnLineThicknessF = 4.0       ; thicker contours
   res15per@cnLineLabelsOn   = False     ; no line labels
   res15per@trGridType = "TriangularMesh"
   res15per@cnLineColor = "black"


   attachres1 = True
   attachres2 = True
   attachres2@gsnAttachPlotsXAxis = True ; attaches along x-axis 

   gsres                   = True                        ; poly res
   gsres@gsFillColor       = "LightGrey"                 ; color chosen
   gsres@gsFillOpacityF = 0.2

   plot_clim_ext = new(2,graphic)
   do is=0,nseasons-1
      plot = new(3,graphic)
      plot_ssmi = new(3,graphic)			; on different grid than HRes so will have to overlay
      plot_JRA = new(3,graphic)                        ; on different grid than HRes so will have to overlay
      poly = new(5,graphic)
      newplot = new(3,graphic)

      plot_clim = new(2,graphic)
      plot_15per = new(6,graphic)
      plot_ext = new(2,graphic)

      if (typeof(seasons).eq."string") then
         season = seasons(is)
         slabel = season_labels(is)
      else
         season = month_names(seasons(is)-1)
         slabel = season
      end if
      psout_season(is) = psout+"."+season
;      wks  = gsn_open_wks("ps",psout_season(is))             ; open a ps file
      wks  = gsn_open_wks("pdf",psout_season(is))             ; open a ps file
;      gsn_define_colormap(wks,"BkBlAqGrYeOrReViWh200")
;      gsn_define_colormap(wks,"cmocean_ice")
;      ltgry    = NhlNewColor(wks,0.7,0.7,0.7)      ; add gray to map
;      dkgry    = NhlNewColor(wks,0.4,0.4,0.4)
;      blk    = NhlNewColor(wks,0.,0.,0.)      ; add gray to map

      resc@gsnPolar  = "SH"
      resc@mpMaxLatF =   -45.
      resc@mpMinLatF =   -90.
      resc@gsnLeftString = "a) SH SIC"

      plot_clim(0) = gsn_csm_contour_map_polar(wks,HR_aice_mn(is,:,:),resc)

      res15per@cnLineColor = "black"
      plot_15per(2) = gsn_csm_contour(wks,HR_aice_mn(is,:,:),res15per)
      overlay(plot_clim(0),plot_15per(2))

      res15per@cnLineColor = "blue"
      plot_15per(0) = gsn_csm_contour(wks,LR_aice_mn(is,:,:),res15per)
      overlay(plot_clim(0),plot_15per(0))

      res15per@cnLineColor = "red"
      plot_15per(1) = gsn_csm_contour(wks,ssmi_mn_sh(is,:,:),res15per)
      overlay(plot_clim(0),plot_15per(1))

      resc@gsnPolar  = "NH"
      resc@mpMaxLatF =   90.
      resc@mpMinLatF =   45.
      resc@gsnLeftString = "b) NH SIC"

      plot_clim(1) = gsn_csm_contour_map_polar(wks,HR_aice_mn(is,:,:),resc)

      res15per@cnLineColor = line_colors(0)
      plot_15per(5) = gsn_csm_contour(wks,HR_aice_mn(is,:,:),res15per)
      overlay(plot_clim(1),plot_15per(5))

      res15per@cnLineColor = line_colors(1)
      plot_15per(3) = gsn_csm_contour(wks,LR_aice_mn(is,:,:),res15per)
      overlay(plot_clim(1),plot_15per(3))

      res15per@cnLineColor = line_colors(2)
      plot_15per(4) = gsn_csm_contour(wks,ssmi_mn_nh(is,:,:),res15per)
      overlay(plot_clim(1),plot_15per(4))

      pres@gsnPanelTop = 0.92
      pres@gsnPanelBottom = 0.56
      pres@txString = "("+HRcase_clim_yr1+"-"+HRcase_clim_yr2+") "+slabel+" mean"
      gsn_panel(wks,plot_clim,(/1,2/),pres)

      res_ext@gsnLeftString = "c) SH "+slabel+" SIE"
      res_ext@tiYAxisString= "10~S~6~N~ km~S~2~N~"
      res_ext@tmYLLabelsOn = True
      res_ext@tmYRLabelsOn = False
      plot_ext(0) = gsn_csm_xy(wks,ext_year,sie_seasonal_sh(:,is,:),res_ext)
      res_ext@gsnLeftString = "d) NH "+slabel+" SIE"
      res_ext@tiYAxisString= ""
      res_ext@tmYLLabelsOn = False
      res_ext@tmYRLabelsOn = True
      plot_ext(1) = gsn_csm_xy(wks,ext_year,sie_seasonal_nh(:,is,:),res_ext)

      pres@gsnPanelLabelBar  = False
      pres@gsnPanelTop = pres@gsnPanelBottom - 0.06
      pres@gsnPanelBottom = 0.29

      if (var2plot.eq."ext") then
         pres@txString = " Sea Ice Extent"
      else
         pres@txString = " Sea Ice Area"
      end if
      gsn_panel(wks,plot_ext,(/1,2/),pres)

      if (is.eq.0) then
         res_clim@gsnLeftString = "e) SH SIE monthly climatology"
         res_clim@tiYAxisString= "10~S~6~N~ km~S~2~N~"
         res_clim@tmYLLabelsOn = True
         res_clim@tmYRLabelsOn = False
         plot_clim_ext(0) = gsn_csm_xy(wks,sie_clim_sh&month,sie_clim_sh,res_clim)
         res_clim@gsnLeftString = "f) NH SIE monthly climatology"
         res_clim@tiYAxisString= ""
         res_clim@tmYLLabelsOn = False
         res_clim@tmYRLabelsOn = True
         plot_clim_ext(1) = gsn_csm_xy(wks,sie_clim_nh&month,sie_clim_nh,res_clim)
      end if
      pres@gsnPanelTop = pres@gsnPanelBottom - 0.04
      pres@gsnPanelBottom = 0.04

      pres@txString = " "
      gsn_panel(wks,plot_clim_ext,(/1,2/),pres)

      if (plot_SH_long) then
         pres@txString = " "


;         res@trYMaxF = 8.
         res@trYMaxF = 9.
         res@trYMinF = 0.

         res_ssmi@gsnLeftString = "Mean Sea Ice Area"
         res_ssmi@tiYAxisString= "10~S~4~N~ km~S~2~N~"
         plot_ssmi(0) = gsn_csm_xy(wks,xSSMI,mn_ifrac_long(is,:),res_ssmi)
         plot_JRA(0) = gsn_csm_xy(wks,xSSMI,mn_JRA_aice_long(is,:),res_JRA)

         if (shade_STD) then
            gsres@gsFillColor       = line_colors(2)
            yp(:nlon-1) = mn_ifrac_long(is,:)+std_ifrac_long(is,:)
            yp(nlon:) = mn_ifrac_long(is,::-1)-std_ifrac_long(is,::-1)
            poly(0) = gsn_add_polygon (wks,plot_ssmi(0),xp,yp,gsres)

            gsres@gsFillColor       = line_colors(1)
            yp(:nlon-1) = mn_JRA_aice_long(is,:)+std_JRA_aice_long(is,:)
            yp(nlon:) = mn_JRA_aice_long(is,::-1)-std_JRA_aice_long(is,::-1)
            poly(1) = gsn_add_polygon (wks,plot_JRA(0),xp,yp,gsres)
         end if

         plot(0) = gsn_csm_xy(wks,xHRes,mn_HRes_aice_long(is,:),res)
         if (shade_STD) then
            gsres@gsFillColor       = line_colors(0)
            yp_HR(:nHR-1) = mn_HRes_aice_long(is,:)+std_HRes_aice_long(is,:)
            yp_HR(nHR:) = mn_HRes_aice_long(is,::-1)-std_HRes_aice_long(is,::-1)
            poly(4) = gsn_add_polygon (wks,plot(0),xp_HR,yp_HR,gsres)
         end if
         overlay(plot(0),plot_ssmi(0))
         overlay(plot(0),plot_JRA(0))

;         res@trYMaxF = 1.8
         res@trYMaxF = 4.
         res@trYMinF = -3.

         res_ssmi@gsnLeftString = "Sea Ice Area STD"
         res_ssmi@tiYAxisString= "(10~S~4~N~ km~S~2~N~)"
         plot_ssmi(1) = gsn_csm_xy(wks,xSSMI,std_ifrac_long(is,:),res_ssmi)
         plot_JRA(1) = gsn_csm_xy(wks,xSSMI,std_JRA_aice_long(is,:),res_JRA)

         plot(1) = gsn_csm_xy(wks,xHRes,std_HRes_aice_long(is,:),res)
         overlay(plot(1),plot_ssmi(1))
         overlay(plot(1),plot_JRA(1))
;         res@trYMaxF = 0.06
;         res@trYMinF = -0.08

         res_ssmi@gsnLeftString = "Sea Ice Area Trends"
         res_ssmi@tiYAxisString= "(10~S~2~N~ km~S~2~N~)yr~S~-1~N~"
         lres@xyLineColor = "gray"
         plot_ssmi(2) = gsn_csm_xy(wks,xSSMI,ifrac_long_trnd(is,:),res_ssmi)
         plot_JRA(2) = gsn_csm_xy(wks,xSSMI,JRA_aice_long_trnd(is,:),res_JRA)

;         res@trYMaxF = 0.06
;         res@trYMinF = -0.08
         plot(2) = gsn_csm_xy(wks,xHRes,HRes_aice_long_trnd(is,:),res)
         dum = gsn_add_polyline(wks,plot(2),(/0.,360./),(/0.,0./),lres)
         overlay(plot(2),plot_ssmi(2))
         overlay(plot(2),plot_JRA(2))

         map1 = gsn_csm_map_ce(wks,resM)
         map2 = gsn_csm_map_ce(wks,resM)
         map3 = gsn_csm_map_ce(wks,resM)
;         newplot(0) = gsn_attach_plots(plot(0),map1,attachres1,attachres2)
         newplot(1) = gsn_attach_plots(plot(1),map2,attachres1,attachres2)
         newplot(2) = gsn_attach_plots(plot(2),map3,attachres1,attachres2)

         pres@gsnPanelTop = pres@gsnPanelBottom - 0.02
         pres@gsnPanelBottom = 0.05
         pres@gsnPanelRight = 0.8

         if (shade_STD) then
            gsn_panel(wks,(/plot(0),plot(2)/),(/2,1/),pres)
         else
            gsn_panel(wks,plot,(/3,1/),pres)
         end if
         xpleg = 0.6
         ypleg = 0.27
         if (shade_STD) then
            ypleg = 0.30
         end if
      else
         xpleg = 0.45
         ypleg = 0.30
      end if
      lgres@lgLineColors       = line_colors
      lgres@lgItemCount        = 3
      lgres@lgItemOrder = (/2,1,0/)
;      labels = (/HRes_short,"JRA 4p2z 002","SSMI "+obs_type/)
      labels = (/"CESM-FOSI-HR","CESM-FOSI-LR","SSMI "+obs_type+" or NSIDC SIE"/)
      gsn_legend_ndc(wks,3,labels,xpleg,ypleg,lgres)

      frame(wks)
      delete(plot)
      delete(newplot)

  end do
end
