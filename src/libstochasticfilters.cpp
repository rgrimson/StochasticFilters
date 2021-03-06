#include "gsl/gsl_sf_psi.h"
#include "gsl/gsl_sf_gamma.h"
#include "gsl/gsl_randist.h"
#include "gsl/gsl_cdf.h"
#include "math.h"
#include <iostream>
#include <vector>
#include <utility>      // std::pair

using namespace std;

/*extern "C" float test_func(float X)
{	
	cout << W1[0].first << ", " << W1[0].second << endl;
	return 0;
}*/

//extern "C" double Gamma_(double x)
//{	
//	cout << "-" << x << "-";
//	return gsl_sf_gamma(x);
//}

inline double Gamma(double x)
{	
	//cout << "-" << x << "-";
	return gsl_sf_gamma(x);
}

//extern "C" double digamma_(double x)
//{	
//	return gsl_sf_psi(x);
//}

inline double digamma(double x)
{	
	return gsl_sf_psi(x);
}


inline double polygamma(int n, double x)
{	
	return gsl_sf_psi_n(n,x);
}

//extern "C" double polygamma_(int n, double x)
//{	
//	return gsl_sf_psi_n(n,x);
//}


inline double nr_l_func(double x)
{
	return log(x)-digamma(x);
}

inline double nr_l_deriv(double x)
{
	return 1/x-polygamma(1,x);
}


extern "C" double NR_L(double y)
{

	double eps = 0.00001; //Newton Raphson tolerance
	double eps2 = 0.004;  //Min value (otherwise answer is too big)
	if(y<eps2)
	{
		//cout << "*"; //<< y << endl;
		return NR_L(eps2);
	}

	double x,f,der;
	x = 0.2;

	f = nr_l_func(x);

	while (f>y)
	{
		x*=2;
		f=nr_l_func(x);
	}
	while (f<y)
	{
		x/=2;
		f=nr_l_func(x);
	}

	while (fabs(f-y)>eps)
	{
		der=nr_l_deriv(x);
        x=x+(y-f)/der;
        f=nr_l_func(x);
	}
        
    return x;

}

#define p(y,x) y*dimx+x


extern "C" void Compute_ML_Param(double M[], double S[],double L[], double m[], int Er, int dimx, int dimy)
{
	double sl, mu;
	int k,x,y,u,v;
	int Wym, WyM, Wxm, WxM;
    cout << "Computing ML parameters (classical windows)" << endl << "Er: " << Er << endl << "(";
    cout << dimx << ")--->" << " ";
    for (x=0;x<dimx;x++)
    	{
    		cout << x << " ";
//    		cout << x << " ";
    		for (y=0;y<dimy;y++)
    		{
	            Wxm=max(0,x-Er);
	            WxM=min(dimx,x+Er+1);
	            Wym=max(0,y-Er);
	            WyM=min(dimy,y+Er+1);
	            mu=0;
	            sl=0;
	            k=0;
	            for (u=Wym;u<WyM;u++)
	            	for (v=Wxm;v<WxM;v++)
	            		{
		            		sl+=log(M[p(u,v)]);
		            		mu+=M[p(u,v)];
		            		k++;
		            	}
	            S[p(y,x)]=mu/k;  //mean over the windows = ML sigma
	            m[p(y,x)]=k;
	            L[p(y,x)]=NR_L(log(S[p(y,x)])-sl/k); //Solve Newton-Raphson = ML number of looks       
    		}
    	}
    cout << "<--- ML" << endl;
}


inline double Wishartpdf(double x,double L,double s)
{
	return gsl_ran_gamma_pdf(x,L,s/L);
}

inline double Chisq(double x, double d)
{
	return gsl_cdf_chisq_P(x,d);
}

typedef vector<pair<int,int>> vp;

class Window {
  public:
    //Window(){};                         // constructor; initialize the list to be empty
    int mx, Mx, my, My;				  // min (m) and max (M) positions
    vp v;

    void compute_min_max()
    {
    	if (v.size()>0);
    	{
    		mx = v[0].second; my = v[0].first; Mx = v[0].second; My = v[0].first;

			for(size_t i=0; i<v.size();i++)
			{
				if( v[i].second < mx)
					mx = v[i].second;
				if( v[i].second >Mx)
					Mx = v[i].second;
				if( v[i].first <my)
					my = v[i].first;
				if( v[i].first > My)
					My = v[i].first;

			}

    	}
    }
};

vp W1={{2,-2},{2,-1},{1,-2},{1,-1},{1,0},{0,-1}};
vp W2={{2,-1},{2,0},{2,1},{1,-1},{1,0},{1,1}};
vp W3={{2,1},{2,2},{1,0},{1,1},{1,2},{0,1}};
vp W4={{-1,-2},{-1,-1},{0,-2},{0,-1},{1,-2},{1,-1}};
vp W5={{-1,-1},{-1,0},{-1,1},{0,-1},{0,1},{1,-1},{1,0},{1,1}};
vp W6={{-1,1},{-1,2},{0,1},{0,2},{1,1},{1,2}};
vp W7={{-2,-2},{-2,-1},{-1,-2},{-1,-1},{-1,0},{0,-1}};
vp W8={{-2,-1},{-2,0},{-2,1},{-1,-1},{-1,0},{-1,1}};
vp W9={{-2,1},{-2,2},{-1,0},{-1,1},{-1,2},{0,1}};
vp WNM_list[9]{W5,W1,W2,W3,W4,W6,W7,W8,W9};

extern "C" void Compute_ML_Param_NMWin(double M[], double S[],double L[], double m[], int dimx, int dimy)
{
	double sl, mu, bestWin=0   ;
	int x,y,cx,cy,coord;
	size_t Ws;
	double th1, th2, p, nth1, nth2, np;

	//deberia ir en un wrapper
	vector<Window> WNM(9,Window());

	for(size_t i=0; i<WNM.size();i++)
	{
		WNM[i].v = WNM_list[i];
		WNM[i].compute_min_max();
	}

		
	Window W;

	//vector<vp> Windows=NMWindows;
    cout << "Computing ML parameters MultiWindows (";
    cout << dimx << ")---> ";
	//cout << WNM.size() << " WNMsize" << endl;

    for (x=0;x<dimx;x++)
    	{
    		cout << x << " ";
    		for (y=0;y<dimy;y++)
    		{
	            th1=0;
	            th2=0;
	            p=-1;
	            nth1=0.1;
	            nth2=1;
	            np=0;

	            for (size_t i=0;i<WNM.size();i++)
	            {
	            	W=WNM[i];

	            	if ((W.mx+x>=0)&(W.Mx+x<dimx)&(W.my+y>=0)&(W.My+y<dimy))
	            	{	
	            		//cout << "x" << i+1 << "(" << y << ", " << x << ") ";
	            		Ws = W.v.size();            	
	            		mu = 0;
			            sl = 0;
			      

			            for (size_t j=0;j<Ws;j++)
		            		{
		            			cx=x+W.v[j].second;
		            			cy=y+W.v[j].first;
		            			coord=cy*dimx+cx;
			            		sl+=log(M[coord]);
			            		mu+=M[coord];
			            	}

			            nth1 = mu/Ws;  //mean over the windows = ML sigma
	                    nth2 = NR_L(log(nth1)-sl/Ws);
	                    np = Wishartpdf(M[p(y,x)],nth2,nth1);

	                    for (size_t j=0;j<Ws;j++)
	                    	{
		            			cx=x+W.v[j].second;
		            			cy=y+W.v[j].first;
		            			coord=cy*dimx+cx;
		            			np*=Wishartpdf(M[coord],nth2,nth1);
		            		}

	                	if (np>p)
	                	{
	                    	p=np;
	                    	bestWin=i;
	                    	//th1=nth1;
	                    	//th2=nth2;
	                    	//m[p(y,x)]=Ws;
	                	}

	            	}
	            }	
	            //cout << WNM.size() << " WNMsize" << endl;
	            //cout << " mu" << th1 << endl;
	            //cout << " lk" << th2 << endl;
                    W=WNM[bestWin];
                    Ws = W.v.size()+1;
                    coord=y*dimx+x;
                    sl=log(M[coord]);
                    mu=M[coord];
                    for (size_t j=0;j<(Ws-1);j++)
                    {
                      cx=x+W.v[j].second;
                      cy=y+W.v[j].first;
                      coord=cy*dimx+cx;
                      sl+=log(M[coord]);
                      mu+=M[coord];
                    }
                    th1 = mu/Ws;  //mean over the windows = ML sigma
                    th2 = NR_L(log(th1)-sl/Ws);

	            S[p(y,x)]=th1;
	            L[p(y,x)]=th2;
                    m[p(y,x)]=Ws;
			    
    		}
    	}
    cout << "<--- ML" << endl;
}

//transforms a p-value into a weight for the convolution kernel
double p2w(double p,double eta)
{
    if (p>=eta)
        return 1;
    if (p<(eta/2))
        return 0;
    //cout<< p << ", ";   
    return (p-eta/2)*2/eta;
}

extern "C" void Filter_fromMLParam(double M[], double F[],double S[],double L[], double m[],int dimx, int dimy, int Sr, double eta, char Filt)
{
	double m1,m2,S1,S2,L1,L2,p,w,sw,St=0;
	//double La; //only for Hellinger
	double l,il,s,gL,dgL,pg1L,v1,v2,h1,h2; //only for Shannon
	double LMaxH=50;
	double LMaxShannon=125.0;
	double Lmin=0.2;

	double dg=2.0; //degrees of freedom for chi2 when using Hellinger or KL
	
	int x,y,xx,yy,coord,Wxm,WxM,Wym,WyM;

	double *H,*V;
	H = new double[dimy*dimx];
	V = new double[dimy*dimx];
     cout << "Sr: " << Sr << endl << "eta: " << eta << endl;

	if (Filt=='K') //Kullback-Leibler
		cout << "Kullback-Leibler Filter --->" << endl;
	else if (Filt=='H')  //Hellinger
		cout << "Hellinger Filter --->" << endl ;
	else if (Filt=='S') //Shannon
	{
		cout << "Shannon Filter --->" << endl;
	    for (x=0;x<dimx;x++)
	  		for (y=0;y<dimy;y++)
	  		{
	  		 l=min(L[p(y,x)],LMaxShannon);
	            il=1.0/l;
	            s=S[p(y,x)];
	            gL=Gamma(l);
	            dgL=digamma(l);
	            pg1L=polygamma(1,l);
	            V[p(y,x)]=pow((1-l)*pg1L+1-il,2)/(pg1L-il)+il;
	            H[p(y,x)]=-log(l)+log(s)+l+(1-l)*dgL+log(gL);	
	  		}
	}
	else //Error: unrecognized Filter!
	{
		cout << "Error: unrecognized Filter!" << endl;
		return;
	}



    for (x=0;x<dimx;x++)
    {
    	cout << x << " ";
  		for (y=0;y<dimy;y++)
  		{
            Wxm=max(0,x-Sr);             //define the Search window limits
            WxM=min(dimx,x+Sr+1);
            Wym=max(0,y-Sr);
            WyM=min(dimy,y+Sr+1);
            sw=0;                        //the sum of the weigths starts in 0
            m1=m[p(y,x)];
            S1=S[p(y,x)];
            L1=L[p(y,x)];
            F[p(y,x)]=0;
            for (xx=Wxm;xx<WxM;xx++)
                for (yy=Wym;yy<WyM;yy++)
                {
                	coord=yy*dimx+xx;
                	//cout<<coord<<" ";
                    m2=m[coord];
                    S2=S[coord];
                    L2=L[coord];
                    if (Filt=='K') //Kullback-Leibler
                    {	
                    	//cout << 'K';
                    	St=m2*m1*(L1+L2)/(m1+m2) * ((S1*S1+S2*S2)/(2*S2*S1)-1); //compute statistics
                      //St=m2*m1/(m1+m2)*(L1-L2)/2 * (log(S1/S2) - log(L1/L2) + digamma(L1) - digamma(L2)) + (L2*S1/S2 + L1*S2/S1)/2 - (L1+L2)/2;
                    	//St=m2*m1/(m1+m2)*((L1-L2) * (log(S1/S2) - log(L1/L2) + digamma(L1) - digamma(L2)) + (L2*S1/S2 + L1*S2/S1) - (L1+L2));
                    	dg=2.0;//degrees of freedom for chi2
                    }
                    else if (Filt=='H')  //Hellinger
                    {
                      if (L1>LMaxH) L1=LMaxH;
                      if (L1<Lmin) L1=Lmin;
                      if (L2>LMaxH) L2=LMaxH;
                      if (L2<Lmin) L2=Lmin;
                    	//cout << 'H';
                    	//La=(L1+L2)/2;
                    	//St=8*m1*m2/(m1+m2)*(1-pow(2,La)*pow(S1*S2,La/2)/pow(S1+S2,La)); //compute statistics
                      //St=4.0*m1*m2/(m1+m2)*(1.0 - Gamma((L1+L2)/2.0) * pow((2/(L1*S2+L2*S1)),(L1+L2)/2) * pow(L1*S2,L1/2) * pow(L2*S1,L2/2) / sqrt((Gamma(L1) * Gamma(L2)) ) );
                    	St=8.0*m1*m2/(m1+m2)*(1.0 - Gamma((L1+L2)/2.0) * pow((2/(L1*S2+L2*S1)),(L1+L2)/2) * pow(L1*S2,L1/2) * pow(L2*S1,L2/2) / sqrt((Gamma(L1) * Gamma(L2)) ) );
                    	dg=2.0;//degrees of freedom for chi2
                    }
                    else if (Filt=='S') //Shannon
                    {
                    	v1=V[p(y,x)];
            			h1=H[p(y,x)];
						v2=V[coord];
            			h2=H[coord];
            			w=1/(m1/v1+m2/v2)*(m1*h1/v1+m2*h2/v2);
                    	St=m1*pow(h1-w,2)/v1+m2*pow(h2-w,2)/v2; //compute statistics
                    	dg=1.0;//degrees of freedom for chi2
                    }
                    p=1-Chisq(St,dg);    //compute p-value
                    w=p2w(p,eta);         //transform p-value in a weight
                    sw+=w;                //sum the weights
                    F[p(y,x)]+=w*M[coord];
                }
            F[p(y,x)]/=sw;
        }
    }

    delete[] V;
    delete[] H;
    cout << "<---" << endl;

}


extern "C" void Filter(double I_Corr[],double I_Filt[],int dimx, int dimy, int Er, int Sr, double eta, char FiltType)
{
	double *S,*L,*m;
	S = new double[dimy*dimx];
	L = new double[dimy*dimx];
	m = new double[dimy*dimx];

	if (Er==0)
	{
		cout << "NMwin" << endl;
		Compute_ML_Param_NMWin( I_Corr, S, L, m, dimx, dimy);
		
	}
	else
	{
		cout << "Center" << endl;
		Compute_ML_Param( I_Corr, S, L, m, Er, dimx, dimy);	
	}
		

	Filter_fromMLParam( I_Corr, I_Filt, S, L, m, dimx, dimy, Sr, eta, FiltType);
	delete[] S;
	delete[] L;
	delete[] m;
}

// extern "C" void Filter_and_Param(double I_Corr[],double I_Filt[],double S[],double L[],double m[],int dimx, int dimy, int Er, int Sr, double eta, char FiltType)
// {
// 	cout << "dimx: " << dimx << "; dimy: " << dimy;
// 	if (Er==0)
// 	{
// 		cout << "NMwin" << endl;
// 		//Compute_ML_Param_NMWin( I_Corr, S, L, m, dimx, dimy);
		
// 	}
// 	else
// 	{
// 		//cout << "Center" << endl;
// 		//Compute_ML_Param( I_Corr, S, L, m, Er, dimx, dimy);	
// 	}
		

// 	//Filter_fromMLParam( I_Corr, S, L, m, I_Filt, dimx, dimy, Sr, eta, FiltType);
// }
