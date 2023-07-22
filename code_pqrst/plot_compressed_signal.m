function plot_compressed_signal(seg, ids)

figure;
plot(seg,'LineWidth',4,'color','b');
for i = 1: length(ids)
    text(ids(i), seg(ids(i)),'o','FontSize',15,'color','r');
end

end