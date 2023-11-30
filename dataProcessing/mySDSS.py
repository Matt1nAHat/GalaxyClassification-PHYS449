import matplotlib.pyplot as plt
from sdss.utils import (decode_objid, sql2df, binimg2array,
                   img_cutout)
from sdss.refs import photo_types

"""
This is a modified version of the PhotoObj class from the objects.py file in the original sdss package.
Minor changes have been made to suit the specific needs of this project, namely changes to the download
and init functions so that additional measurements could be pulled from the database to create our feature 
vector. As such the code in this file will not be commented as it is not original work.
"""

class PhotoObj:
    def __init__(self, objID):
        
        self.objID = str(objID)
        
        dc = decode_objid(objID)
        self.sky_version = dc['version']
        self.rerun = dc['rerun']
        self.run = dc['run']
        self.camcol = dc['camcol']
        self.field = dc['field']
        self.id_in_field = dc['id_within_field']
        
        self.specObjID = None
        self.ra = None
        self.dec = None
        self.mag = None
        self.type = None
        self.dist2sel = None
        self.downloaded = False

        #Additional measurement initializations added (From Matthew Charbonneau)
        self.fiberColour = None
        self.modelColour = None
        self.petroColour = None
        self.petroR50 = None
        self.petroR90 = None
        self.secondMoment = None
        self.fourthMoment = None
        self.axisRatioDEV = None
        self.axisRatioEXP = None
        self.ellipticityE1 = None
        self.ellipticityE2 = None
        self.modelFitDEV = None
        self.modelFitEXP = None
        self.modelFitSTAR = None
        self.p_el = None
        self.p_cw = None
        self.p_acw = None
        self.p_edge = None
        self.p_mg = None

    def download(self, get_image=False):
        #Script has been updated to pull additional measurements from the database (From Matthew Charbonneau)
        script = """
        SELECT p.specObjID, p.ra, p.dec, p.u, p.g, p.r, p.i, p.z, p.type, p.deVAB_u, p.deVAB_g, p.deVAB_r, p.deVAB_i, p.deVAB_z, p.expAB_u, p.expAB_g, p.expAB_r, p.expAB_i, p.expAB_z,
            p.lnLstar_u, p.lnLstar_g, p.lnLstar_r, p.lnLstar_i, p.lnLstar_z, p.lnLdeV_u, p.lnLdeV_g, p.lnLdeV_r, p.lnLdeV_i, p.lnLdeV_z, p.lnLexp_u, p.lnLexp_g, p.lnLexp_r,
            p.lnLexp_i, p.lnLexp_z, p.mE2_u, p.mE2_g, p.mE2_r, p.mE2_i, p.mE2_z, p.mE1_u, p.mE1_g, p.mE1_r, p.mE1_i, p.mE1_z, p.mRrCc_u, p.mRrCc_g, p.mRrCc_r, p.mRrCc_i, p.mRrCc_z, p.mCr4_u,
            p.mCr4_g, p.mCr4_r, p.mCr4_i, p.mCr4_z, p.fiberMag_u, p.fiberMag_g, p.fiberMag_r, p.fiberMag_i, p.fiberMag_z, p.modelMag_u, p.modelMag_g, p.modelMag_r, p.modelMag_i,
            p.modelMag_z, p.petroMag_u, p.petroMag_g, p.petroMag_r, p.petroMag_i, p.petroMag_z, p.petroR50_u, p.petroR50_g, p.petroR50_r, p.petroR50_i, p.petroR50_z, p.petroR90_u,
            p.petroR90_g, p.petroR90_r, p.petroR90_i, p.petroR90_z, zns.nvote, zns.p_el as elliptical, zns.p_cw as spiralclock, zns.p_acw as spiralanticlock, 
            zns.p_edge as edgeon, zns.p_mg as merger
        """

        script = script + f"FROM PhotoObjAll as p JOIN ZooNoSpec AS zns ON p.objID=zns.objid AND p.objID={self.objID}"
        df = sql2df(script)
        if len(df)>0:
            float_cols = ['ra','dec','u','g','r','i','z','deVAB_u','deVAB_g','deVAB_r','deVAB_i','deVAB_z','expAB_u','expAB_g','expAB_r',
                          'expAB_i','expAB_z','lnLstar_u','lnLstar_g','lnLstar_r','lnLstar_i','lnLstar_z','lnLdeV_u','lnLdeV_g','lnLdeV_r',
                          'lnLdeV_i','lnLdeV_z','mE2_u','mE2_g','mE2_r','mE2_i','mE2_z','mE1_u','mE1_g','mE1_r','mE1_i','mE1_z','mRrCc_u',
                          'mRrCc_g','mRrCc_r','mRrCc_i','mRrCc_z','mCr4_u','mCr4_g','mCr4_r','mCr4_i','mCr4_z','fiberMag_u','fiberMag_g',
                          'fiberMag_r','fiberMag_i','fiberMag_z','modelMag_u','modelMag_g','modelMag_r','modelMag_i','modelMag_z','petroMag_u',
                          'petroMag_g','petroMag_r','petroMag_i','petroMag_z','petroR50_u','petroR50_g','petroR50_r','petroR50_i','petroR50_z',
                          'petroR90_u','petroR90_g','petroR90_r','petroR90_i','petroR90_z','nvote','elliptical','spiralclock','spiralanticlock','edgeon','merger']
            
            df[float_cols] = df[float_cols].astype(float)
            self.specObjID = df['specObjID'].iloc[0]
            self.ra = df['ra'].iloc[0]
            self.dec = df['dec'].iloc[0]
            u = df['u'].iloc[0]
            g = df['g'].iloc[0]
            r = df['r'].iloc[0]
            i = df['i'].iloc[0]
            z = df['z'].iloc[0]
            self.mag = {'u':u, 'g':g, 'r':r, 'i':i, 'z':z}
            #Assign additional measurements (From Matthew Charbonneau)
            self.fiberColour = {'u':df['fiberMag_u'].iloc[0], 'g':df['fiberMag_g'].iloc[0], 'r':df['fiberMag_r'].iloc[0], 'i':df['fiberMag_i'].iloc[0], 'z':df['fiberMag_z'].iloc[0]}
            self.modelColour = {'u':df['modelMag_u'].iloc[0], 'g':df['modelMag_g'].iloc[0], 'r':df['modelMag_r'].iloc[0], 'i':df['modelMag_i'].iloc[0], 'z':df['modelMag_z'].iloc[0]}
            self.petroColour = {'u':df['petroMag_u'].iloc[0], 'g':df['petroMag_g'].iloc[0], 'r':df['petroMag_r'].iloc[0], 'i':df['petroMag_i'].iloc[0], 'z':df['petroMag_z'].iloc[0]}       
            self.petroR50 = {'u':df['petroR50_u'].iloc[0], 'g':df['petroR50_g'].iloc[0], 'r':df['petroR50_r'].iloc[0], 'i':df['petroR50_i'].iloc[0], 'z':df['petroR50_z'].iloc[0]}
            self.petroR90 = {'u':df['petroR90_u'].iloc[0], 'g':df['petroR90_g'].iloc[0], 'r':df['petroR90_r'].iloc[0], 'i':df['petroR90_i'].iloc[0], 'z':df['petroR90_z'].iloc[0]}
            self.secondMoment = {'u':df['mRrCc_u'].iloc[0], 'g':df['mRrCc_g'].iloc[0], 'r':df['mRrCc_r'].iloc[0], 'i':df['mRrCc_i'].iloc[0], 'z':df['mRrCc_z'].iloc[0]}
            self.fourthMoment = {'u':df['mCr4_u'].iloc[0], 'g':df['mCr4_g'].iloc[0], 'r':df['mCr4_r'].iloc[0], 'i':df['mCr4_i'].iloc[0], 'z':df['mCr4_z'].iloc[0]}
            self.axisRatioDEV = {'u':df['deVAB_u'].iloc[0], 'g':df['deVAB_g'].iloc[0], 'r':df['deVAB_r'].iloc[0], 'i':df['deVAB_i'].iloc[0], 'z':df['deVAB_z'].iloc[0]}
            self.axisRatioEXP = {'u':df['expAB_u'].iloc[0], 'g':df['expAB_g'].iloc[0], 'r':df['expAB_r'].iloc[0], 'i':df['expAB_i'].iloc[0], 'z':df['expAB_z'].iloc[0]}
            self.ellipticityE1 = {'u':df['mE1_u'].iloc[0], 'g':df['mE1_g'].iloc[0], 'r':df['mE1_r'].iloc[0], 'i':df['mE1_i'].iloc[0], 'z':df['mE1_z'].iloc[0]}
            self.ellipticityE2 = {'u':df['mE2_u'].iloc[0], 'g':df['mE2_g'].iloc[0], 'r':df['mE2_r'].iloc[0], 'i':df['mE2_i'].iloc[0], 'z':df['mE2_z'].iloc[0]}
            self.modelFitDEV = {'u':df['lnLdeV_u'].iloc[0], 'g':df['lnLdeV_g'].iloc[0], 'r':df['lnLdeV_r'].iloc[0], 'i':df['lnLdeV_i'].iloc[0], 'z':df['lnLdeV_z'].iloc[0]}
            self.modelFitEXP = {'u':df['lnLexp_u'].iloc[0], 'g':df['lnLexp_g'].iloc[0], 'r':df['lnLexp_r'].iloc[0], 'i':df['lnLexp_i'].iloc[0], 'z':df['lnLexp_z'].iloc[0]}
            self.modelFitSTAR = {'u':df['lnLstar_u'].iloc[0], 'g':df['lnLstar_g'].iloc[0], 'r':df['lnLstar_r'].iloc[0], 'i':df['lnLstar_i'].iloc[0], 'z':df['lnLstar_z'].iloc[0]}
            self.p_el = df['elliptical'].iloc[0]
            self.p_cw = df['spiralclock'].iloc[0]
            self.p_acw = df['spiralanticlock'].iloc[0]
            self.p_edge = df['edgeon'].iloc[0]
            self.p_mg = df['merger'].iloc[0]

            self.type = photo_types[df['type'].iloc[0]]
            if get_image:
                self.img_array = self.quick_image()
        self.downloaded = True

    def cutout_image(self, scale=0.1, width=300, height=300):
        if not self.downloaded:
            self.download()
        if self.ra is None:
            raise Exception('Photo object not found!')
        data = img_cutout(ra=self.ra, dec=self.dec, scale=scale,
                          width=width, height=height, opt='', query='')
        return data

    def show(self, scale=0.1, width=200, height=200):
        data = self.cutout_image(scale=scale, width=width, height=height)
        plt.imshow(data)
        plt.axis('off') # new
        plt.show()
