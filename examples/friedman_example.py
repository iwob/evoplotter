from src.stats import friedman

sample = """;$GP$_$0.01$;$GP$_$0.1$;CDGP_$0.01$;CDGP_$0.1$;$CDGP_{props}$_$0.01$;$CDGP_{props}$_$0.1$
gr_b_05;0.00;0.00;0.76;0.80;1.00;1.00
gr_m_05;0.00;0.00;0.00;0.08;0.96;0.92
gr_s_05;0.12;0.16;0.84;0.92;0.96;1.00
gr_bm_05;0.00;0.00;0.00;0.04;0.84;0.96
gr_bs_05;0.12;0.08;0.84;1.00;1.00;1.00
gr_ms_05;0.00;0.00;0.00;0.04;0.84;0.96
gr_bms_05;0.00;0.00;0.00;0.04;1.00;1.00
res2_c1_05;0.00;0.08;0.60;0.48;0.76;0.72
res2_c2_05;0.00;0.04;0.08;0.44;0.80;0.84
res2_s_05;0.00;0.00;0.36;0.68;0.96;1.00
res2_sc_05;0.00;0.04;0.08;0.32;0.96;0.76
res3_c1_05;0.00;0.00;0.08;0.04;0.28;0.36
res3_c2_05;0.00;0.04;0.00;0.04;0.16;0.16
res3_s_05;0.00;0.00;0.60;0.52;0.72;0.68
res3_sc_05;0.00;0.00;0.00;0.00;0.24;0.12"""

sample2 = """;$m0.25, c0.75$;$m0.5, c0.5$;$m0.75, c0.25$;$m1.0, c0.0$
CountPositive2;0.94;0.95;0.94;0.94
CountPositive3;0.73;0.68;0.48;0.34
Median3;0.99;0.96;0.95;0.81
SortedAscending4;0.81;0.84;0.84;0.84
SortedAscending5;0.71;0.76;0.78;0.65
fg_max4;0.99;0.99;0.98;0.84
"""

# sample3 is the same as sample2, but prepared so that p-value is significant
sample3 = """;$m0.25, c0.75$;$m0.5, c0.5$;$m0.75, c0.25$;$m1.0, c0.0$
CountPositive2;0.00;0.95;0.94;0.94
CountPositive3;0.00;0.68;0.48;0.34
Median3;0.00;0.96;0.95;0.81
SortedAscending4;0.00;0.84;0.84;0.84
SortedAscending5;0.00;0.76;0.78;0.65
fg_max4;0.00;0.99;0.98;0.84
"""


sample4 = """;$GPR$_$m0.25,c0.75$;$GPR$_$m0.5,c0.5$;$GPR$_$m0.75,c0.25$;$GPR$_$m1.0,c0.0$;CDGP_$0.75$_$m0.25,c0.75$;CDGP_$0.75$_$m0.5,c0.5$;CDGP_$0.75$_$m0.75,c0.25$;CDGP_$0.75$_$m1.0,c0.0$;CDGP_$1.0$_$m0.25,c0.75$;CDGP_$1.0$_$m0.5,c0.5$;CDGP_$1.0$_$m0.75,c0.25$;CDGP_$1.0$_$m1.0,c0.0$;$CDGP_{props}$_$1$_$0.75$_$m0.25,c0.75$;$CDGP_{props}$_$1$_$0.75$_$m0.5,c0.5$;$CDGP_{props}$_$1$_$0.75$_$m0.75,c0.25$;$CDGP_{props}$_$1$_$0.75$_$m1.0,c0.0$;$CDGP_{props}$_$1$_$1.0$_$m0.25,c0.75$;$CDGP_{props}$_$1$_$1.0$_$m0.5,c0.5$;$CDGP_{props}$_$1$_$1.0$_$m0.75,c0.25$;$CDGP_{props}$_$1$_$1.0$_$m1.0,c0.0$
CountPositive2;0.79;0.80;0.77;0.74;0.98;1.00;1.00;1.00;0.85;0.87;0.89;0.96;1.00;1.00;1.00;1.00;1.00;1.00;1.00;0.98
CountPositive3;0.13;0.10;0.06;0.06;0.95;0.97;1.00;1.00;0.40;0.43;0.13;0.19;0.99;0.98;0.91;0.25;0.94;0.97;0.83;0.00
Median3;0.92;0.83;0.78;0.53;1.00;1.00;1.00;0.98;0.98;0.91;0.92;0.83;1.00;1.00;1.00;0.96;1.00;1.00;0.99;0.76
SortedAscending4;0.37;0.44;0.38;0.47;1.00;0.98;1.00;1.00;0.76;0.67;0.75;0.78;0.93;0.96;0.99;0.97;0.87;0.94;0.93;0.91
SortedAscending5;0.00;0.05;0.04;0.00;0.88;0.98;1.00;1.00;0.62;0.67;0.72;0.62;0.91;0.93;1.00;0.98;0.88;0.93;0.94;0.62
fg_max4;0.96;0.96;0.84;0.74;1.00;1.00;1.00;1.00;1.00;1.00;1.00;0.92;1.00;1.00;1.00;0.93;1.00;1.00;1.00;0.33
"""

fRes = friedman.runFriedmanKK_csv(sample4)

print("\nRESULT:")
# print("\n".join([str(x) for x in fRes.getSignificantPairs()]))
print(fRes.getSignificantPairsText())