% Re-plotter for workspaces

close all;

yline(testErrorKRR,'black','LineWidth',2);
hold on;
errorbar((1:maximumMRFF),mean(testErrorComplexRFF),std(testErrorComplexRFF),'LineWidth',2);
hold on
errorbar((1:maximumRankFF)*D*M,mean(testErrorFull),std(testErrorFull),'LineWidth',2);
hold on
errorbar(2*(1:maximumRankQFF)*D*log2(M),mean(testErrorQuant),std(testErrorQuant),'LineWidth',2);
hold on
xline(N,'--','LineWidth',2);
hold off
xlabel('$P$','interpreter','latex','FontSize',20);
ylabel('MSE','interpreter','latex','FontSize',20);
high = max(max(max(testErrorComplexRFF(:)),max(testErrorFull(:))),max(max(testErrorQuant(:)),testErrorKRR(:)));
low = min(min(min(testErrorComplexRFF(:)),min(testErrorFull(:))),min(min(testErrorQuant(:)),testErrorKRR(:)));
xlim([min(M*D,2*D*log2(M)),maximumMRFF]);
xticklabels('auto');
legend('KRR','RFF','TKM','QTKM','interpreter','latex','FontSize',15);
axis square;
set(gca,'XScale','log','YScale','log','LineWidth',1.5);
title('qsar\_fish','interpreter','latex','FontSize',20)
subtitle('$N=908$, $D=6$','interpreter','latex','FontSize',20)
exportgraphics(gca,"qsarFishTest.pdf");