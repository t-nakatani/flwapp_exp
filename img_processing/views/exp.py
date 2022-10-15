from django.contrib.auth.decorators import login_required
from django.http.response import HttpResponse

@login_required
def register_img(request):
    """実験の前準備として初回の花弁配置値の推定を完了して保存するための機能"""

    if request.method == 'POST':
        pass
        # form = ApplicationForm(request.POST)
        # if form.is_valid():
        #     applicaion = form.save(commit=False)
        #     applicaion.worker = request.user
        #     applicaion.work = get_object_or_404(Work, pk=work_id)
        #     applicaion.save()
        #     return redirect('home')
    else:
        if request.user.id == 1:
            return HttpResponse('user==nakatani OK')
        # work = get_object_or_404(Work, pk=work_id)
        # if Application.objects.filter(work=work, worker=request.user).exists:
        #     return render(request, 'work/apply.html')
        # form = ApplicationForm()
        # return render(request, 'work/apply.html', {'form': form})

